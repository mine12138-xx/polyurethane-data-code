from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors, rdPartialCharges


ROOT = Path(__file__).resolve().parents[1]
DENSITY_FEATURES = [
    "NumHAcceptors", "NumHDonors", "NOCount", "fr_NH1", "fr_NH2", "fr_CO", "fr_ether",
    "NumNHCO", "CH3_groups", "TPSA", "LabuteASA", "PEOE_VSA8", "VSA_EState8",
    "NumRotatableBonds", "RingCount", "HeavyAtomCount", "log_Vm",
]
MOLAR_AVERAGE_FEATURES = [
    "MPC", "BCUT2D_MRHI", "FractionCSP3", "BertzCT", "FpDensityMorgan3", "MolLogP", "Qed", "SPS",
]
MASS_AVERAGE_FEATURES = ["ROG", "PMI", "Ecc", "Acc", "ISF"]
MOLECULAR_WEIGHT_FEATURES = ["MolWt", "HeavyAtomMolWt"]
SMARTS_PATTERNS = {
    "NumNHCO": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
    "CH3_groups": Chem.MolFromSmarts("[CX4H3]"),
}


def extract_base_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_3d = Chem.AddHs(mol)
    has_3d = False
    try:
        if AllChem.EmbedMolecule(mol_3d, randomSeed=42) == 0:
            AllChem.MMFFOptimizeMolecule(mol_3d)
            has_3d = True
    except Exception:
        pass
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = [
        float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
        for i in range(mol.GetNumAtoms())
        if mol.GetAtomWithIdx(i).HasProp("_GasteigerCharge")
    ]
    features = {
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NOCount": Descriptors.NOCount(mol),
        "fr_NH1": Descriptors.fr_NH1(mol),
        "fr_NH2": Descriptors.fr_NH2(mol),
        "fr_CO": Descriptors.fr_C_O(mol),
        "fr_ether": Descriptors.fr_ether(mol),
        "NumNHCO": len(mol.GetSubstructMatches(SMARTS_PATTERNS["NumNHCO"])),
        "CH3_groups": len(mol.GetSubstructMatches(SMARTS_PATTERNS["CH3_groups"])),
        "TPSA": Descriptors.TPSA(mol),
        "LabuteASA": Descriptors.LabuteASA(mol),
        "PEOE_VSA8": Descriptors.PEOE_VSA8(mol),
        "VSA_EState8": Descriptors.VSA_EState8(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "log_Vm": np.log10(AllChem.ComputeMolVolume(mol_3d)) if has_3d else np.log10(Descriptors.MolWt(mol) / 1.1),
        "MPC": max([abs(charge) for charge in charges]) if charges else 0,
        "BCUT2D_MRHI": Descriptors.BCUT2D_MRHI(mol) if hasattr(Descriptors, "BCUT2D_MRHI") else 0,
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "FpDensityMorgan3": Descriptors.FpDensityMorgan3(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "Qed": QED.qed(mol),
        "SPS": rdMolDescriptors.CalcSPS(mol) if hasattr(rdMolDescriptors, "CalcSPS") else 0.0,
    }
    if has_3d:
        features.update({
            "ROG": rdMolDescriptors.CalcRadiusOfGyration(mol_3d),
            "PMI": rdMolDescriptors.CalcPMI1(mol_3d),
            "Ecc": rdMolDescriptors.CalcEccentricity(mol_3d),
            "Acc": rdMolDescriptors.CalcAsphericity(mol_3d),
            "ISF": rdMolDescriptors.CalcInertialShapeFactor(mol_3d),
        })
    else:
        features.update({feature: 0.0 for feature in MASS_AVERAGE_FEATURES})
    features["MolWt_SMILES"] = Descriptors.MolWt(mol)
    features["HeavyAtomMolWt"] = Descriptors.HeavyAtomMolWt(mol)
    features["MolWt"] = Descriptors.MolWt(mol)
    features["CED"] = 0.0
    return features


def calculate_segment_features(components, base_db):
    if not components:
        return {}
    total_moles = sum(component["moles"] for component in components)
    total_mass = sum(component["moles"] * component["table_mw"] for component in components)
    if total_moles == 0 or total_mass == 0:
        return {}
    segment_features = {}
    for feature in DENSITY_FEATURES:
        value_sum = sum(
            base_db[component["name"]][feature] / base_db[component["name"]]["MolWt_SMILES"]
            * component["moles"] * component["table_mw"]
            for component in components
        )
        segment_features[f"{feature}_Density"] = value_sum / total_mass
    for feature in MOLAR_AVERAGE_FEATURES:
        value_sum = sum(base_db[component["name"]][feature] * component["moles"] for component in components)
        segment_features[feature] = value_sum / total_moles
    for feature in MASS_AVERAGE_FEATURES:
        value_sum = sum(
            base_db[component["name"]][feature] * component["moles"] * component["table_mw"]
            for component in components
        )
        segment_features[feature] = value_sum / total_mass
    for feature in MOLECULAR_WEIGHT_FEATURES:
        if feature == "MolWt":
            segment_features[feature] = total_mass / total_moles
        else:
            value_sum = sum(
                base_db[component["name"]]["HeavyAtomMolWt"]
                * (component["table_mw"] / base_db[component["name"]]["MolWt_SMILES"])
                * component["moles"]
                for component in components
            )
            segment_features[feature] = value_sum / total_moles
    segment_features["CED"] = 0.0
    return segment_features


def run_pu_pipeline(smiles_path, formulation_path, output_path):
    df_smiles = pd.read_csv(smiles_path) if smiles_path.endswith(".csv") else pd.read_excel(smiles_path)
    df_form = pd.read_csv(formulation_path) if formulation_path.endswith(".csv") else pd.read_excel(formulation_path)
    if df_smiles["Abbreviation"].duplicated().any():
        raise ValueError("Component abbreviations must be unique.")
    print("Calculating base descriptors for the SMILES library...")
    base_db = {row["Abbreviation"]: extract_base_features(row["SMILES"]) for _, row in df_smiles.iterrows()}
    mw_dict = dict(zip(df_smiles["Abbreviation"], df_smiles["Molecular_weight"]))
    results = []
    print("Applying segment definitions and mixture-weighted descriptor calculations...")
    for _, row in df_form.iterrows():
        formulation = row["Formulation"]
        soft_name = row["Soft_segment"] if pd.notna(row["Soft_segment"]) else None
        isocyanate_name = row["Diisocyanate"] if pd.notna(row["Diisocyanate"]) else None
        extender_1 = row["Chain_extender_1"] if pd.notna(row["Chain_extender_1"]) else None
        extender_2 = row["Chain_extender_2"] if pd.notna(row["Chain_extender_2"]) else None
        ratios = [float(value) for value in str(row["Molar_ratio"]).split(":")]
        components = [
            {"name": soft_name, "moles": ratios[0]} if len(ratios) > 0 and soft_name else None,
            {"name": isocyanate_name, "moles": ratios[1]} if len(ratios) > 1 and isocyanate_name else None,
            {"name": extender_1, "moles": ratios[2]} if len(ratios) > 2 and extender_1 else None,
            {"name": extender_2, "moles": ratios[3]} if len(ratios) > 3 and extender_2 else None,
        ]
        valid_components = []
        for component in components:
            if component and component["name"] in mw_dict:
                component["table_mw"] = mw_dict[component["name"]]
                valid_components.append(component)
            elif component:
                raise ValueError(f"Unknown component {component['name']} in formulation {formulation}")
        soft_features = calculate_segment_features([c for c in valid_components if c["name"] == soft_name], base_db)
        hard_features = calculate_segment_features(
            [c for c in valid_components if c["name"] in [isocyanate_name, extender_1, extender_2]], base_db
        )
        all_features = calculate_segment_features(valid_components, base_db)
        formulation_result = {"Formulation": formulation}
        formulation_result.update({f"SS_{key}": value for key, value in soft_features.items()})
        formulation_result.update({f"HS_{key}": value for key, value in hard_features.items()})
        formulation_result.update({f"PU_{key}": value for key, value in all_features.items()})
        results.append(formulation_result)
    output = pd.DataFrame(results)
    output.to_csv(output_path, index=False)
    print(f"Calculated {output.shape[1] - 1} descriptor columns for {len(output)} formulations.")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract descriptors for virtual formulations.")
    parser.add_argument("--smiles", type=Path, default=ROOT / "data/raw/SMILES.xlsx")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/Virtual_PU_Fixed.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/Virtual_Features.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_pu_pipeline(str(args.smiles), str(args.input), str(args.output))


if __name__ == "__main__":
    main()
