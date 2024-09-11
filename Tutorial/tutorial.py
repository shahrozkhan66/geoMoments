import numpy as np
from stl import mesh
import pyvista as pv
from geometricOperators.geometric_operators import *



import pandas as pd
def main():
    # Define the risk register data
    data = {
        "ID": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
        "Risk Description": [
            "Environmental impact due to computational resource usage",
            "Commercial viability linked to WindWings success",
            "Stakeholder engagement and confidence",
            "Project execution and delivery",
            "Data security breaches and unauthorized access",
            "Performance degradation with up to 100 concurrent users",
            "Dependencies on cloud services (InfluxDB, MongoDB, AWS)",
            "Custom security solutions risks"
        ],
        "Likelihood": ["Low", "Medium", "Medium", "Medium", "High", "Medium", "Medium", "Low"],
        "Impact": ["Low", "High", "High", "High", "High", "Medium", "Medium", "High"],
        "Mitigation Strategy": [
            "Follow BAR Technology's strategy to minimize environmental impact of computational resources",
            "Leverage WindWings' strong track record and stakeholder interest; communicate regularly with stakeholders",
            "Maintain effective communication and regular updates to stakeholders",
            "Utilize team's experience in complex project delivery; conduct regular reviews and progress checks",
            "Implement AWS Cognito for authentication, RBAC, TLS for data encryption, stringent API endpoint checks, regular data backups",
            "Configure monitoring to alert for performance issues, ensure prompt resolution, use AWS infrastructure for scalability",
            "Ensure robust SLAs with cloud providers, regular monitoring, and contingency planning",
            "Adhere to industry best practices, avoid custom security solutions, use proven security frameworks"
        ],
        "Owner": ["Project Manager", "Commercial Lead", "Project Manager", "Project Manager", "Security Lead", "Technical Lead", "Technical Lead", "Security Lead"],
        "Status": ["Open", "Open", "Open", "Open", "Open", "Open", "Open", "Open"]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    file_path = r"H:\152 DM-BAR tanker optimisation - General\C- Naval Arch\Dimension Reduction\01 Smart Shipping Grant\risk_register.csv"
    df.to_csv(file_path, index=False)

    file_path



    design_path = Path("hull_mesh_0.stl")
    stl_data = mesh.Mesh.from_file(design_path)

    gm = GeometricOperator(stl_data)
    operators = []
    names = []
    for order in range(6):
        operator_, names_ = gm.getOperatorVector(order, 0, 0)  # order, isCentered, isScaled
        operators.extend(operator_)
        names.extend(names_)

    print(f"Volume (Zeroth Order) = {operators[0]}")  # the zeroth order operator (m_000) is volume
    print(
        f"centeriod: c_x {operators[1] / operators[0]} c_y {operators[2] / operators[0]} c_z {operators[3] / operators[0]}")
    print(f"First Order: x {operators[1]} y {operators[2]} z {operators[3]}")
    print(f"Second Order: xx {operators[4]} yy {operators[6]} zz {operators[9]}")

    print("\nFrom Rhino: Volume = 279628.299")
    print("From Rhino: Volume Centroid = c_x 44.595173, c_y -6.50838774, c_z 28.4422043")
    print("From Rhino: First Moments = x 12470072.4, y -1819929.39, z 7953245.2")
    print("From Rhino: Second Moments = xx 708343887, yy 77164264.7, zz 395148046")


if __name__ == '__main__':
    main()
