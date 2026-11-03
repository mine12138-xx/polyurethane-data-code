from pathlib import Path
import argparse
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_CHAIN_EXTENDERS = {"FBA", "OBB", "ABB", "PBA"}
COMPONENT_COLUMNS = [
    "Soft_segment",
    "Soft_segment_1",
    "Soft_segment_2",
    "Diisocyanate",
    "Diisocyanate_1",
    "Diisocyanate_2",
    "Chain_extender",
    "Chain_extender_1",
    "Chain_extender_2",
    "Chain_extender_3",
]


def fix_virtual_library_ratios_csv(input_csv, output_csv):
    print("Reading virtual-library data...")
    try:
        df = pd.read_csv(input_csv, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding="gbk")

    comp_cols = [column for column in COMPONENT_COLUMNS if column in df.columns]
    fixed_count = 0
    for index, row in df.iterrows():
        ratio_str = str(row.get("Molar_ratio", ""))
        if pd.isna(ratio_str) or ratio_str.strip() in {"", "nan"}:
            continue
        ratios = [value.strip() for value in re.split(r"[:：]", ratio_str) if value.strip()]
        current_comps = []
        current_comp_types = []
        for column in comp_cols:
            value = str(row[column]).strip()
            if value and value != "nan":
                current_comps.append(value)
                current_comp_types.append("CE" if "Chain_extender" in column or column == "Chain_extender" else "Other")
        if len(ratios) != len(current_comps):
            continue

        modified = False
        new_ratios = ratios.copy()
        recipe_name = str(row.get("Formulation", ""))
        for i, (component, component_type) in enumerate(zip(current_comps, current_comp_types)):
            if component_type == "CE" and component in TARGET_CHAIN_EXTENDERS:
                try:
                    if float(new_ratios[i]) == 1.0:
                        new_ratios[i] = "0.66"
                        modified = True
                        pattern = re.compile(re.escape(component) + r"\(\s*1(?:\.0+)?\s*\)")
                        recipe_name = pattern.sub(f"{component}(0.66)", recipe_name)
                except ValueError:
                    continue
        if modified:
            df.at[index, "Molar_ratio"] = ":".join(new_ratios)
            if "Formulation" in df.columns:
                df.at[index, "Formulation"] = recipe_name
            fixed_count += 1

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Adjusted {fixed_count} formulations containing FBA, OBB, ABB, or PBA.")
    print(f"Saved repaired data to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Apply the documented trimer ratio representation rule.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/Virtual_PU.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/Virtual_PU_Fixed.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fix_virtual_library_ratios_csv(str(args.input), str(args.output))


if __name__ == "__main__":
    main()
