from pathlib import Path
import argparse
import itertools

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def generate_virtual_pu_library_2411(input_excel, output_csv):
    print("Loading molecular building-block data...")
    try:
        df = pd.read_excel(input_excel)
    except Exception as exc:
        print(f"Could not read the Excel file; check the path: {exc}")
        raise

    required = {"Abbreviation", "Role", "SMILES", "Molecular_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["Molecular_weight"] = pd.to_numeric(df["Molecular_weight"], errors="coerce")
    df = df.dropna(subset=["Molecular_weight"])
    soft_segments = df[df["Role"].astype(str).str.contains("soft segment", case=False, na=False)]
    isocyanates = df[df["Role"].astype(str).str.contains("diisocyanate", case=False, na=False)]
    chain_extenders = df[df["Role"].astype(str).str.contains("chain extender", case=False, na=False)]
    print(
        f"Found {len(soft_segments)} soft segments, {len(isocyanates)} diisocyanates, "
        f"and {len(chain_extenders)} chain extenders."
    )
    print("Enumerating all 2:4:1:1 formulations with two chain-extender positions...")

    virtual_library = []
    ce_pairs = list(itertools.combinations_with_replacement(chain_extenders.to_dict("records"), 2))
    total_expected = len(soft_segments) * len(isocyanates) * len(ce_pairs)
    print(
        f"Expected formulations: {len(soft_segments)} x {len(isocyanates)} x "
        f"{len(ce_pairs)} = {total_expected}"
    )
    for _, soft in soft_segments.iterrows():
        for _, iso in isocyanates.iterrows():
            for ce1, ce2 in ce_pairs:
                n_soft, n_iso, n_ce1, n_ce2 = 2, 4, 1, 1
                mass_soft = n_soft * soft["Molecular_weight"]
                mass_hard = (
                    n_iso * iso["Molecular_weight"]
                    + n_ce1 * ce1["Molecular_weight"]
                    + n_ce2 * ce2["Molecular_weight"]
                )
                total_mass = mass_soft + mass_hard
                hsc = mass_hard / total_mass
                if ce1["Abbreviation"] == ce2["Abbreviation"]:
                    recipe_name = (
                        f"{soft['Abbreviation']}(2)-{iso['Abbreviation']}(4)-"
                        f"{ce1['Abbreviation']}(2)"
                    )
                else:
                    recipe_name = (
                        f"{soft['Abbreviation']}(2)-{iso['Abbreviation']}(4)-"
                        f"{ce1['Abbreviation']}(1)-{ce2['Abbreviation']}(1)"
                    )
                virtual_library.append(
                    {
                        "Formulation": recipe_name,
                        "Soft_segment": soft["Abbreviation"],
                        "Diisocyanate": iso["Abbreviation"],
                        "Chain_extender_1": ce1["Abbreviation"],
                        "Chain_extender_2": ce2["Abbreviation"],
                        "Molar_ratio": "2:4:1:1",
                        "Hard_segment_content_HSC": round(hsc, 4),
                        "Soft_segment_SMILES": soft["SMILES"],
                        "Diisocyanate_SMILES": iso["SMILES"],
                        "Chain_extender_1_SMILES": ce1["SMILES"],
                        "Chain_extender_2_SMILES": ce2["SMILES"],
                        "Total_molecular_weight_per_repeat_unit": round(total_mass, 2),
                    }
                )
    result_df = pd.DataFrame(virtual_library)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Generated {len(result_df)} virtual polyurethane formulations.")
    print(f"Saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Enumerate the virtual polyurethane library.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/SMILES.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/Virtual_PU.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_virtual_pu_library_2411(str(args.input), str(args.output))


if __name__ == "__main__":
    main()
