import shutil
from pathlib import Path

import libcst as cst

from freqtrade.constants import Config


class StrategyUpdater:
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
        # Strategy/config settings:
        "use_sell_signal": "use_exit_signal",
        "sell_profit_only": "exit_profit_only",
        "sell_profit_offset": "exit_profit_offset",
        "ignore_roi_if_buy_signal": "ignore_roi_if_entry_signal",
        "forcebuy_enable": "force_entry_enable",
    }
    rename_dict = {
        "buy": "entry",
        "sell": "exit",
        "buy_tag": "entry_tag",
    }
    function_mapping = {
        "populate_buy_trend": "populate_entry_trend",
        "populate_sell_trend": "populate_exit_trend",
        "custom_sell": "custom_exit",
        "check_buy_timeout": "check_entry_timeout",
        "check_sell_timeout": "check_exit_timeout",
    }

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

    def update_code(self, code: str) -> str:
        tree = cst.parse_module(code)
        updated_tree = tree.visit(NameUpdater())
        return updated_tree.code


class NameUpdater(cst.CSTTransformer):
    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in original_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == "INTERFACE_VERSION":
                return updated_node.with_changes(value=cst.Integer("3"))
        return updated_node

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        if any(
                isinstance(base.value, cst.Name) and base.value.value == "IStrategy"
                for base in original_node.bases
        ):
            has_interface_version = False
            statements = list(updated_node.body.body)
            for stmt in statements:
                if (
                        isinstance(stmt, cst.SimpleStatementLine)
                        and len(stmt.body) == 1
                        and isinstance(stmt.body[0], cst.Assign)
                ):
                    for target in stmt.body[0].targets:
                        if (
                                isinstance(target.target, cst.Name)
                                and target.target.value == "INTERFACE_VERSION"
                        ):
                            has_interface_version = True
                            break
                if has_interface_version:
                    break

            if not has_interface_version:
                new_line = cst.SimpleStatementLine(
                    body=[cst.Assign(
                        targets=[cst.AssignTarget(cst.Name("INTERFACE_VERSION"))],
                        value=cst.Integer("3"))
                    ]
                )
                statements.insert(0, new_line)
                return updated_node.with_changes(
                    body=updated_node.body.with_changes(body=tuple(statements))
                )
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        if original_node.value in StrategyUpdater.name_mapping:
            return updated_node.with_changes(value=StrategyUpdater.name_mapping[original_node.value])
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.name.value in StrategyUpdater.function_mapping:
            return updated_node.with_changes(
                name=cst.Name(StrategyUpdater.function_mapping[original_node.name.value])
            )
        return updated_node

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.Attribute:
        if (
                isinstance(original_node.value, cst.Name)
                and original_node.value.value == "trade"
                and original_node.attr.value == "nr_of_successful_buys"
        ):
            return updated_node.with_changes(attr=cst.Name("nr_of_successful_entries"))
        return updated_node

    def leave_Dict(self, original_node: cst.Dict, updated_node: cst.Dict) -> cst.Dict:
        new_elements = []
        for element in original_node.elements:
            new_key = element.key
            new_value = element.value
            if isinstance(element.key, cst.SimpleString):
                raw_key = element.key.evaluated_value.strip("\"'")
                mapped_key = StrategyUpdater.rename_dict.get(raw_key, raw_key)
                if raw_key != mapped_key:
                    new_key = element.key.with_changes(value=f"'{mapped_key}'")
            if isinstance(element.value, cst.SimpleString):
                raw_value = element.value.evaluated_value.strip("\"'")
                new_value = element.value.with_changes(value=f"'{raw_value}'")
            new_elements.append(element.with_changes(key=new_key, value=new_value))
        return updated_node.with_changes(elements=new_elements)

    def leave_Subscript(self, original_node: cst.Subscript, updated_node: cst.Subscript) -> cst.Subscript:
        new_slices = []
        for slice_elem in original_node.slice:
            if isinstance(slice_elem.slice, cst.Index):
                index_value = slice_elem.slice.value
                if isinstance(index_value, cst.SimpleString):
                    key = index_value.evaluated_value.strip("\"'")
                    if key in StrategyUpdater.name_mapping:
                        new_slices.append(
                            slice_elem.with_changes(
                                slice=cst.Index(value=cst.SimpleString(f"'{StrategyUpdater.name_mapping[key]}'"))
                            )
                        )
                    else:
                        new_slices.append(slice_elem)
                elif isinstance(index_value, cst.List):
                    new_elements = []
                    for element in index_value.elements:
                        if isinstance(element.value, cst.SimpleString):
                            old_str = element.value.evaluated_value.strip("\"'")
                            new_str = StrategyUpdater.name_mapping.get(old_str, old_str)
                            new_elements.append(cst.Element(cst.SimpleString(f"'{new_str}'")))
                        else:
                            new_elements.append(element)
                    new_list = cst.Index(value=cst.List(new_elements))
                    new_slices.append(slice_elem.with_changes(slice=new_list))
                else:
                    new_slices.append(slice_elem)
            else:
                new_slices.append(slice_elem)
        return updated_node.with_changes(slice=tuple(new_slices))

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.Comparison:
        new_comparisons = []
        for comp in original_node.comparisons:
            if isinstance(comp.operator, cst.Equal) and isinstance(comp.comparator, cst.SimpleString):
                key = comp.comparator.evaluated_value.strip("\"'")
                if key in StrategyUpdater.name_mapping:
                    new_comparisons.append(
                        comp.with_changes(
                            comparator=cst.SimpleString(f"'{StrategyUpdater.name_mapping[key]}'")
                        )
                    )
                else:
                    new_comparisons.append(comp)
            else:
                new_comparisons.append(comp)
        return updated_node.with_changes(comparisons=new_comparisons)
