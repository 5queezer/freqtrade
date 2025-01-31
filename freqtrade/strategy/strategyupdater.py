import shutil
from pathlib import Path

import libcst as cst

from freqtrade.constants import Config


class StrategyUpdater:
    def start(self, config: Config, strategy_obj: dict) -> None:
        source_file = strategy_obj["location"]
        strategies_backup_folder = Path(config["user_data_dir"]) / "strategies_orig_updater"
        target_file = strategies_backup_folder / strategy_obj["location_rel"]
        with open(source_file, "r") as f:
            old_code = f.read()
        if not strategies_backup_folder.exists():
            strategies_backup_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_file, target_file)
        new_code = self.update_code(old_code)
        with open(source_file, "w") as f:
            f.write(new_code)

    @staticmethod
    def update_code(code: str) -> str:
        tree = cst.parse_module(code)
        updated_tree = tree.visit(NameUpdater())
        return updated_tree.code


class NameUpdater(cst.CSTTransformer):
    """ Applies necessary updates to strategy code while preserving formatting and comments. """

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        """ Ensures `INTERFACE_VERSION = 3` is set inside strategy classes. """
        for target in original_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == "INTERFACE_VERSION":
                return updated_node.with_changes(value=cst.Integer("3"))
        return updated_node

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """ Inserts `INTERFACE_VERSION = 3` inside strategy classes if missing. """
        if any(isinstance(base.value, cst.Name) and base.value.value == "IStrategy" for base in original_node.bases):
            statements = list(updated_node.body.body)
            if not any(
                    isinstance(stmt, cst.SimpleStatementLine)
                    and isinstance(stmt.body[0], cst.Assign)
                    and any(isinstance(t.target, cst.Name) and t.target.value == "INTERFACE_VERSION"
                            for t in stmt.body[0].targets)
                    for stmt in statements
            ):
                statements.insert(0, cst.SimpleStatementLine(
                    body=[cst.Assign(
                        targets=[cst.AssignTarget(cst.Name("INTERFACE_VERSION"))],
                        value=cst.Integer("3"))
                    ]
                ))
                return updated_node.with_changes(body=updated_node.body.with_changes(body=tuple(statements)))
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """ Renames functions and updates parameters. """
        function_mapping = {
            "populate_buy_trend": "populate_entry_trend",
            "populate_sell_trend": "populate_exit_trend",
            "custom_sell": "custom_exit",
            "check_buy_timeout": "check_entry_timeout",
            "check_sell_timeout": "check_exit_timeout",
        }
        requires_side = original_node.name.value in {"custom_stake_amount", "confirm_trade_entry", "custom_entry_price"}

        param_list = []
        for param in original_node.params.params:
            new_name = cst.Name("exit_reason") if param.name.value == "sell_reason" else param.name
            new_annotation = param.annotation

            if new_annotation and isinstance(new_annotation.annotation, cst.Subscript):
                subscript = new_annotation.annotation
                if isinstance(subscript.value, cst.Name) and subscript.value.value == "Optional":
                    if isinstance(subscript.slice[0].slice, cst.Index):
                        inner_type = subscript.slice[0].slice.value
                        new_annotation = new_annotation.with_changes(
                            annotation=cst.BinaryOperation(
                                left=inner_type,
                                operator=cst.BitOr(),
                                right=cst.Name("None")
                            )
                        )

            param_list.append(param.with_changes(name=new_name, annotation=new_annotation))

        if requires_side:
            side_param = cst.Param(
                name=cst.Name("side"),
                annotation=cst.Annotation(cst.Name("str"))
            )
            if param_list and param_list[-1].name.value == "kwargs":
                param_list.insert(-1, side_param)
            else:
                param_list.append(side_param)

        new_function_name = function_mapping.get(original_node.name.value, original_node.name.value)
        return updated_node.with_changes(name=cst.Name(new_function_name),
                                         params=updated_node.params.with_changes(params=param_list))

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        """ Renames variables and strategy attributes. """
        name_mapping = {
            "ticker_interval": "timeframe",
            "buy": "enter_long",
            "sell": "exit_long",
            "buy_tag": "enter_tag",
            "sell_reason": "exit_reason",
            "sell_signal": "exit_signal",
            "custom_sell": "custom_exit",
            "force_sell": "force_exit",
            "emergency_sell": "emergency_exit",
            "use_sell_signal": "use_exit_signal",
            "sell_profit_only": "exit_profit_only",
            "sell_profit_offset": "exit_profit_offset",
            "ignore_roi_if_buy_signal": "ignore_roi_if_entry_signal",
            "forcebuy_enable": "force_entry_enable",
        }
        return updated_node.with_changes(value=name_mapping.get(original_node.value, original_node.value))

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.Attribute:
        """ Updates trade object attributes. """
        if original_node.value.value == "trade" and original_node.attr.value == "nr_of_successful_buys":
            return updated_node.with_changes(attr=cst.Name("nr_of_successful_entries"))
        return updated_node

    def leave_Dict(self, original_node: cst.Dict, updated_node: cst.Dict) -> cst.Dict:
        """ Updates dictionary keys while preserving formatting. """
        rename_dict = {"buy": "entry", "sell": "exit", "buy_tag": "entry_tag"}
        new_elements = []
        for element in original_node.elements:
            new_key = element.key
            if isinstance(element.key, cst.SimpleString):
                raw_key = element.key.evaluated_value.strip("\"'")
                new_key = element.key.with_changes(
                    value=f"{element.key.quote}{rename_dict.get(raw_key, raw_key)}{element.key.quote}")
            new_elements.append(element.with_changes(key=new_key))
        return updated_node.with_changes(elements=new_elements)

    def leave_Subscript(self, original_node: cst.Subscript, updated_node: cst.Subscript) -> cst.Subscript:
        """ Updates DataFrame column names inside `dataframe[...]` subscripts. """
        name_mapping = {
            "buy": "enter_long",
            "sell": "exit_long",
            "buy_tag": "enter_tag"
        }

        new_slices = []
        for slice_elem in original_node.slice:
            if isinstance(slice_elem.slice, cst.Index):
                slice_value = slice_elem.slice.value
                if isinstance(slice_value, cst.List):
                    new_elements = []
                    for element in slice_value.elements:
                        if isinstance(element.value, cst.SimpleString):
                            key = element.value.evaluated_value.strip("\"'")
                            new_key = name_mapping.get(key, key)
                            new_elements.append(
                                element.with_changes(value=cst.SimpleString(
                                    f"{element.value.quote}{new_key}{element.value.quote}"
                                ))
                            )
                        else:
                            new_elements.append(element)

                    new_slices.append(slice_elem.with_changes(
                        slice=cst.Index(value=cst.List(elements=new_elements))
                    ))
                elif isinstance(slice_value, cst.SimpleString):
                    key = slice_value.evaluated_value.strip("\"'")
                    new_key = name_mapping.get(key, key)
                    new_slices.append(slice_elem.with_changes(
                        slice=cst.Index(value=cst.SimpleString(
                            f"{slice_value.quote}{new_key}{slice_value.quote}"
                        ))
                    ))
                else:
                    new_slices.append(slice_elem)
            else:
                new_slices.append(slice_elem)

        return updated_node.with_changes(slice=tuple(new_slices))

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.Comparison:
        """ Updates comparison expressions like `sell_reason == "x"` """
        name_mapping = {"sell_signal": "exit_signal", "force_sell": "force_exit", "emergency_sell": "emergency_exit"}
        new_comparisons = []
        for comp in original_node.comparisons:
            if isinstance(comp.operator, cst.Equal) and isinstance(comp.comparator, cst.SimpleString):
                key = comp.comparator.evaluated_value.strip("\"'")
                new_comparisons.append(comp.with_changes(comparator=cst.SimpleString(
                    f"{comp.comparator.quote}{name_mapping.get(key, key)}{comp.comparator.quote}")))
            else:
                new_comparisons.append(comp)
        return updated_node.with_changes(comparisons=new_comparisons)
