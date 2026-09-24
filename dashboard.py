import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

#Title of the dashboard
st.title("⚙️ Predictive Maintenance System")

st.write(
    """
    This dashboard uses machine learning to predict machine failure
    from sensor parameters and provide a maintenance recommendation.
    """
)

MODEL_PATH = (
    "models/predictive_maintenance_model.pkl"
)

DATASET_PATH = "ai4i2020.csv"

if not os.path.exists(MODEL_PATH):

    st.error(
        """
        Trained model not found.

        Please run:

        python predictive_maintenance.py
        """
    )

    st.stop()


with open(MODEL_PATH, "rb") as file:

    model = pickle.load(file)

if not os.path.exists(DATASET_PATH):

    st.error(
        "ai4i2020.csv was not found."
    )

    st.stop()


df = pd.read_csv(
    DATASET_PATH
)
DATASET_RANGES = {

    "Air temperature [K]": (
        df["Air temperature [K]"].min(),
        df["Air temperature [K]"].max()
    ),

    "Process temperature [K]": (
        df["Process temperature [K]"].min(),
        df["Process temperature [K]"].max()
    ),

    "Rotational speed [rpm]": (
        df["Rotational speed [rpm]"].min(),
        df["Rotational speed [rpm]"].max()
    ),

    "Torque [Nm]": (
        df["Torque [Nm]"].min(),
        df["Torque [Nm]"].max()
    ),

    "Tool wear [min]": (
        df["Tool wear [min]"].min(),
        df["Tool wear [min]"].max()
    )
}
type_mapping = {
    "L": 0,
    "M": 1,
    "H": 2
}

def maintenance_recommendation(
    failure_probability,
    tool_wear,
    air_temperature,
    process_temperature,
    torque,
    rotational_speed
):

    warnings = []

    temperature_difference = (
        process_temperature - air_temperature
    )

    if temperature_difference > 10:

        warnings.append(
            "High temperature difference detected."
        )

    if torque > 50:

        warnings.append(
            "High torque detected."
        )

    if rotational_speed < 1200:

        warnings.append(
            "Low rotational speed detected."
        )

    if tool_wear >= 200:

        warnings.append(
            "Critical tool wear."
        )

    elif tool_wear >= 150:

        warnings.append(
            "High tool wear."
        )

    elif tool_wear >= 100:

        warnings.append(
            "Moderate tool wear."
        )


    if (
        failure_probability >= 0.75
        or tool_wear >= 200
    ):

        status = (
            "IMMEDIATE MAINTENANCE REQUIRED"
        )

        estimated_hours = 0


    elif (
        failure_probability >= 0.50
        or tool_wear >= 150
    ):

        status = (
            "MAINTENANCE REQUIRED SOON"
        )

        estimated_hours = 10


    elif (
        failure_probability >= 0.25
        or tool_wear >= 100
    ):

        status = (
            "MAINTENANCE SHOULD BE SCHEDULED"
        )

        estimated_hours = 25


    else:

        status = (
            "MACHINE CONDITION NORMAL"
        )

        estimated_hours = 50


    return (
        status,
        estimated_hours,
        warnings
    )

st.header("🔧 Machine Sensor Inputs")

st.write(
    "Enter the current machine operating parameters."
)


col1, col2, col3 = st.columns(3)


# Machine Type
with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    st.caption(
        "L = Low, M = Medium, H = High"
    )


# Air Temperature
with col2:

    air_min, air_max = (
        DATASET_RANGES[
            "Air temperature [K]"
        ]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=float(
            round(air_min, 1)
        ),
        max_value=float(
            round(air_max, 1)
        ),
        value=float(
            round(
                df[
                    "Air temperature [K]"
                ].mean(),
                1
            )
        ),
        step=0.1
    )

    st.caption(
        f"Dataset range: {air_min:.1f} - {air_max:.1f} K"
    )


# Process Temperature
with col3:

    process_min, process_max = (
        DATASET_RANGES[
            "Process temperature [K]"
        ]
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=float(
            round(process_min, 1)
        ),
        max_value=float(
            round(process_max, 1)
        ),
        value=float(
            round(
                df[
                    "Process temperature [K]"
                ].mean(),
                1
            )
        ),
        step=0.1
    )

    st.caption(
        f"Dataset range: {process_min:.1f} - {process_max:.1f} K"
    )


col4, col5, col6 = st.columns(3)


# Rotational Speed
with col4:

    speed_min, speed_max = (
        DATASET_RANGES[
            "Rotational speed [rpm]"
        ]
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=float(
            round(speed_min, 0)
        ),
        max_value=float(
            round(speed_max, 0)
        ),
        value=float(
            round(
                df[
                    "Rotational speed [rpm]"
                ].mean(),
                0
            )
        ),
        step=1.0
    )

    st.caption(
        f"Dataset range: {speed_min:.0f} - {speed_max:.0f} rpm"
    )


# Torque
with col5:

    torque_min, torque_max = (
        DATASET_RANGES[
            "Torque [Nm]"
        ]
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=float(
            round(torque_min, 1)
        ),
        max_value=float(
            round(torque_max, 1)
        ),
        value=float(
            round(
                df[
                    "Torque [Nm]"
                ].mean(),
                1
            )
        ),
        step=0.1
    )

    st.caption(
        f"Dataset range: {torque_min:.1f} - {torque_max:.1f} Nm"
    )


# Tool Wear
with col6:

    wear_min, wear_max = (
        DATASET_RANGES[
            "Tool wear [min]"
        ]
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=float(
            round(wear_min, 0)
        ),
        max_value=float(
            round(wear_max, 0)
        ),
        value=float(
            round(
                df[
                    "Tool wear [min]"
                ].mean(),
                0
            )
        ),
        step=1.0
    )

    st.caption(
        f"Dataset range: {wear_min:.0f} - {wear_max:.0f} min"
    )

st.subheader("Current Sensor Values")

sensor_table = pd.DataFrame({

    "Parameter": [
        "Machine Type",
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ],

    "Value": [
        machine_type,
        f"{air_temperature:.1f} K",
        f"{process_temperature:.1f} K",
        f"{rotational_speed:.0f} rpm",
        f"{torque:.1f} Nm",
        f"{tool_wear:.0f} min"
    ]
})

st.dataframe(
    sensor_table,
    use_container_width=True,
    hide_index=True
)

st.divider()

predict_button = st.button(
    "🔍 Predict Machine Condition",
    type="primary",
    use_container_width=True
)

if predict_button:

    type_value = type_mapping[
        machine_type
    ]

    input_data = pd.DataFrame(

        [[
            type_value,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear
        ]],

        columns=[
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    )


    # Prediction
    prediction = model.predict(
        input_data
    )[0]


    # Probability
    probability = model.predict_proba(
        input_data
    )[0][1]


    # Maintenance
    status, estimated_hours, warnings = (
        maintenance_recommendation(
            probability,
            tool_wear,
            air_temperature,
            process_temperature,
            torque,
            rotational_speed
        )
    )

    st.header("📊 Prediction Result")

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ MACHINE FAILURE DETECTED"
            )

        else:

            st.success(
                "✅ MACHINE CONDITION NORMAL"
            )


    with result_col2:

        st.metric(
            "Failure Probability",
            f"{probability * 100:.2f}%"
        )


    with result_col3:

        if estimated_hours == 0:

            st.metric(
                "Maintenance",
                "IMMEDIATE"
            )

        else:

            st.metric(
                "Maintenance Estimate",
                f"{estimated_hours} hours"
            )
    st.subheader(
        "🔧 Maintenance Assessment"
    )

    if (
        status ==
        "IMMEDIATE MAINTENANCE REQUIRED"
    ):

        st.error(status)

    elif (
        status ==
        "MAINTENANCE REQUIRED SOON"
    ):

        st.warning(status)

    elif (
        status ==
        "MAINTENANCE SHOULD BE SCHEDULED"
    ):

        st.info(status)

    else:

        st.success(status)

    st.subheader(
        "⚠️ Sensor Warnings"
    )

    if len(warnings) == 0:

        st.success(
            "No major sensor warning detected."
        )

    else:

        for warning in warnings:

            st.warning(
                warning
            )

    st.subheader(
        "Failure Probability"
    )

    probability_data = pd.DataFrame({

        "Condition": [
            "No Failure",
            "Failure"
        ],

        "Probability": [
            1 - probability,
            probability
        ]
    })

    st.bar_chart(
        probability_data.set_index(
            "Condition"
        )
    )

    st.subheader(
        "Tool Wear Analysis"
    )

    wear_data = pd.DataFrame({

        "Tool Wear": [
            tool_wear
        ]

    })

    st.bar_chart(
        wear_data
    )

st.divider()

st.header("📈 Dataset Analysis")


tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "Failure Analysis",
    "Sensor Analysis",
    "Failure Types",
    "Statistics",
    "Dataset Information"

])


# TAB 1 - FAILURE ANALYSIS

with tab1:

    st.subheader(
        "Machine Failure Distribution"
    )

    failure_count = (
        df["Machine failure"]
        .value_counts()
        .sort_index()
    )

    failure_chart = pd.DataFrame({

        "Condition": [
            "No Failure",
            "Failure"
        ],

        "Machines": [
            failure_count.get(0, 0),
            failure_count.get(1, 0)
        ]
    })

    st.bar_chart(
        failure_chart.set_index(
            "Condition"
        )
    )


# ============================================================
# PREDICTIVE MAINTENANCE DASHBOARD
# STREAMLIT WEB APPLICATION
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚙️ Predictive Maintenance System")

st.write(
    """
    This dashboard uses machine learning to predict machine failure
    from sensor parameters and provide a maintenance recommendation.
    """
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = (
    "model/predictive_maintenance_model.pkl"
)

DATASET_PATH = "ai4i2020.csv"


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        """
        Trained model not found.

        Please run:

        python predictive_maintenance.py
        """
    )

    st.stop()


with open(MODEL_PATH, "rb") as file:

    model = pickle.load(file)


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    st.error(
        "ai4i2020.csv was not found."
    )

    st.stop()


df = pd.read_csv(
    DATASET_PATH
)


# ============================================================
# DATASET RANGES
# ============================================================

DATASET_RANGES = {

    "Air temperature [K]": (
        df["Air temperature [K]"].min(),
        df["Air temperature [K]"].max()
    ),

    "Process temperature [K]": (
        df["Process temperature [K]"].min(),
        df["Process temperature [K]"].max()
    ),

    "Rotational speed [rpm]": (
        df["Rotational speed [rpm]"].min(),
        df["Rotational speed [rpm]"].max()
    ),

    "Torque [Nm]": (
        df["Torque [Nm]"].min(),
        df["Torque [Nm]"].max()
    ),

    "Tool wear [min]": (
        df["Tool wear [min]"].min(),
        df["Tool wear [min]"].max()
    )
}


# ============================================================
# MACHINE TYPE
# ============================================================

type_mapping = {
    "L": 0,
    "M": 1,
    "H": 2
}


# ============================================================
# MAINTENANCE FUNCTION
# ============================================================

def maintenance_recommendation(
    failure_probability,
    tool_wear,
    air_temperature,
    process_temperature,
    torque,
    rotational_speed
):

    warnings = []

    temperature_difference = (
        process_temperature - air_temperature
    )

    if temperature_difference > 10:

        warnings.append(
            "High temperature difference detected."
        )

    if torque > 50:

        warnings.append(
            "High torque detected."
        )

    if rotational_speed < 1200:

        warnings.append(
            "Low rotational speed detected."
        )

    if tool_wear >= 200:

        warnings.append(
            "Critical tool wear."
        )

    elif tool_wear >= 150:

        warnings.append(
            "High tool wear."
        )

    elif tool_wear >= 100:

        warnings.append(
            "Moderate tool wear."
        )


    if (
        failure_probability >= 0.75
        or tool_wear >= 200
    ):

        status = (
            "IMMEDIATE MAINTENANCE REQUIRED"
        )

        estimated_hours = 0


    elif (
        failure_probability >= 0.50
        or tool_wear >= 150
    ):

        status = (
            "MAINTENANCE REQUIRED SOON"
        )

        estimated_hours = 10


    elif (
        failure_probability >= 0.25
        or tool_wear >= 100
    ):

        status = (
            "MAINTENANCE SHOULD BE SCHEDULED"
        )

        estimated_hours = 25


    else:

        status = (
            "MACHINE CONDITION NORMAL"
        )

        estimated_hours = 50


    return (
        status,
        estimated_hours,
        warnings
    )


# ============================================================
# SECTION 1 - MACHINE INPUT
# ============================================================

st.header("🔧 Machine Sensor Inputs")

st.write(
    "Enter the current machine operating parameters."
)


col1, col2, col3 = st.columns(3)


# Machine Type
with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    st.caption(
        "L = Low, M = Medium, H = High"
    )


# Air Temperature
with col2:

    air_min, air_max = (
        DATASET_RANGES[
            "Air temperature [K]"
        ]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=float(
            round(air_min, 1)
        ),
        max_value=float(
            round(air_max, 1)
        ),
        value=float(
            round(
                df[
                    "Air temperature [K]"
                ].mean(),
                1
            )
        ),
        step=0.1
    )

    st.caption(
        f"Dataset range: {air_min:.1f} - {air_max:.1f} K"
    )


# Process Temperature
with col3:

    process_min, process_max = (
        DATASET_RANGES[
            "Process temperature [K]"
        ]
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=float(
            round(process_min, 1)
        ),
        max_value=float(
            round(process_max, 1)
        ),
        value=float(
            round(
                df[
                    "Process temperature [K]"
                ].mean(),
                1
            )
        ),
        step=0.1
    )

    st.caption(
        f"Dataset range: {process_min:.1f} - {process_max:.1f} K"
    )


col4, col5, col6 = st.columns(3)


# Rotational Speed
with col4:

    speed_min, speed_max = (
        DATASET_RANGES[
            "Rotational speed [rpm]"
        ]
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=float(
            round(speed_min, 0)
        ),
        max_value=float(
            round(speed_max, 0)
        ),
        value=float(
            round(
                df[
                    "Rotational speed [rpm]"
                ].mean(),
                0
            )
        ),
        step=1.0
    )

    st.caption(
        f"Dataset range: {speed_min:.0f} - {speed_max:.0f} rpm"
    )


# Torque
with col5:

    torque_min, torque_max = (
        DATASET_RANGES[
            "Torque [Nm]"
        ]
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=float(
            round(torque_min, 1)
        ),
        max_value=float(
            round(torque_max, 1)
        ),
        value=float(
            round(
                df[
                    "Torque [Nm]"
                ].mean(),
                1
            )
        ),
        step=0.1
    )

    st.caption(
        f"Dataset range: {torque_min:.1f} - {torque_max:.1f} Nm"
    )


# Tool Wear
with col6:

    wear_min, wear_max = (
        DATASET_RANGES[
            "Tool wear [min]"
        ]
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=float(
            round(wear_min, 0)
        ),
        max_value=float(
            round(wear_max, 0)
        ),
        value=float(
            round(
                df[
                    "Tool wear [min]"
                ].mean(),
                0
            )
        ),
        step=1.0
    )

    st.caption(
        f"Dataset range: {wear_min:.0f} - {wear_max:.0f} min"
    )


# ============================================================
# SENSOR INPUT TABLE
# ============================================================

st.subheader("Current Sensor Values")

sensor_table = pd.DataFrame({

    "Parameter": [
        "Machine Type",
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ],

    "Value": [
        machine_type,
        f"{air_temperature:.1f} K",
        f"{process_temperature:.1f} K",
        f"{rotational_speed:.0f} rpm",
        f"{torque:.1f} Nm",
        f"{tool_wear:.0f} min"
    ]
})

st.dataframe(
    sensor_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Machine Condition",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    type_value = type_mapping[
        machine_type
    ]

    input_data = pd.DataFrame(

        [[
            type_value,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear
        ]],

        columns=[
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    )


    # Prediction
    prediction = model.predict(
        input_data
    )[0]


    # Probability
    probability = model.predict_proba(
        input_data
    )[0][1]


    # Maintenance
    status, estimated_hours, warnings = (
        maintenance_recommendation(
            probability,
            tool_wear,
            air_temperature,
            process_temperature,
            torque,
            rotational_speed
        )
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.header("📊 Prediction Result")

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ MACHINE FAILURE DETECTED"
            )

        else:

            st.success(
                "✅ MACHINE CONDITION NORMAL"
            )


    with result_col2:

        st.metric(
            "Failure Probability",
            f"{probability * 100:.2f}%"
        )


    with result_col3:

        if estimated_hours == 0:

            st.metric(
                "Maintenance",
                "IMMEDIATE"
            )

        else:

            st.metric(
                "Maintenance Estimate",
                f"{estimated_hours} hours"
            )


    # ========================================================
    # MAINTENANCE STATUS
    # ========================================================

    st.subheader(
        "🔧 Maintenance Assessment"
    )

    if (
        status ==
        "IMMEDIATE MAINTENANCE REQUIRED"
    ):

        st.error(status)

    elif (
        status ==
        "MAINTENANCE REQUIRED SOON"
    ):

        st.warning(status)

    elif (
        status ==
        "MAINTENANCE SHOULD BE SCHEDULED"
    ):

        st.info(status)

    else:

        st.success(status)


    # ========================================================
    # SENSOR WARNINGS
    # ========================================================

    st.subheader(
        "⚠️ Sensor Warnings"
    )

    if len(warnings) == 0:

        st.success(
            "No major sensor warning detected."
        )

    else:

        for warning in warnings:

            st.warning(
                warning
            )


    # ========================================================
    # FAILURE PROBABILITY GRAPH
    # ========================================================

    st.subheader(
        "Failure Probability"
    )

    probability_data = pd.DataFrame({

        "Condition": [
            "No Failure",
            "Failure"
        ],

        "Probability": [
            1 - probability,
            probability
        ]
    })

    st.bar_chart(
        probability_data.set_index(
            "Condition"
        )
    )


    # ========================================================
    # TOOL WEAR GRAPH
    # ========================================================

    st.subheader(
        "Tool Wear Analysis"
    )

    wear_data = pd.DataFrame({

        "Tool Wear": [
            tool_wear
        ]

    })

    st.bar_chart(
        wear_data
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

st.divider()

st.header("📈 Dataset Analysis")


tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "Failure Analysis",
    "Sensor Analysis",
    "Failure Types",
    "Statistics",
    "Dataset Information"

])


# ============================================================
# TAB 1 - FAILURE ANALYSIS
# ============================================================

with tab1:

    st.subheader(
        "Machine Failure Distribution"
    )

    failure_count = (
        df["Machine failure"]
        .value_counts()
        .sort_index()
    )

    failure_chart = pd.DataFrame({

        "Condition": [
            "No Failure",
            "Failure"
        ],

        "Machines": [
            failure_count.get(0, 0),
            failure_count.get(1, 0)
        ]
    })

    st.bar_chart(
        failure_chart.set_index(
            "Condition"
        )
    )


# TAB 2 - SENSOR ANALYSIS

with tab2:

    st.subheader(
        "Sensor Parameter Statistics"
    )

    sensor_columns = [

        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"

    ]

    sensor_statistics = df[
        sensor_columns
    ].describe().T

    st.dataframe(
        sensor_statistics,
        use_container_width=True
    )

# TAB 3 - FAILURE TYPES
# ============================================================

with tab3:

    st.subheader(
        "Failure Type Analysis"
    )

    failure_columns = [
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    failure_names = {

        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure"

    }

    failure_type_data = pd.DataFrame({

        "Failure Code": failure_columns,

        "Failure Type": [
            failure_names[x]
            for x in failure_columns
        ],

        "Number of Cases": [
            int(df[x].sum())
            for x in failure_columns
        ]

    })

    st.dataframe(
        failure_type_data,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        failure_type_data.set_index(
            "Failure Code"
        )["Number of Cases"]
    )


# ============================================================
# TAB 4 - STATISTICS
# ============================================================

with tab4:

    st.subheader(
        "Statistical Analysis"
    )

    statistical_columns = [

        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"

    ]

    statistics_table = pd.DataFrame({

        "Minimum": [
            df[column].min()
            for column in statistical_columns
        ],

        "Maximum": [
            df[column].max()
            for column in statistical_columns
        ],

        "Mean": [
            df[column].mean()
            for column in statistical_columns
        ],

        "Median": [
            df[column].median()
            for column in statistical_columns
        ],

        "Standard Deviation": [
            df[column].std()
            for column in statistical_columns
        ],

        "Variance": [
            df[column].var()
            for column in statistical_columns
        ]

    }, index=statistical_columns)

    st.dataframe(
        statistics_table,
        use_container_width=True
    )


# ============================================================
# TAB 5 - DATASET INFORMATION
# ============================================================

with tab5:

    st.subheader(
        "Dataset Information"
    )

    info_col1, info_col2, info_col3 = (
        st.columns(3)
    )

    with info_col1:

        st.metric(
            "Number of Rows",
            df.shape[0]
        )

    with info_col2:

        st.metric(
            "Number of Columns",
            df.shape[1]
        )

    with info_col3:

        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )


    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

# ============================================================
# INPUT VARIABLE RANGES
# ============================================================

st.divider()

st.header(
    "📏 Input Variable Ranges"
)

range_table = pd.DataFrame({

    "Variable": [
        "Machine Type",
        "Air Temperature [K]",
        "Process Temperature [K]",
        "Rotational Speed [rpm]",
        "Torque [Nm]",
        "Tool Wear [min]"
    ],

    "Minimum": [
        "L",
        DATASET_RANGES[
            "Air temperature [K]"
        ][0],

        DATASET_RANGES[
            "Process temperature [K]"
        ][0],

        DATASET_RANGES[
            "Rotational speed [rpm]"
        ][0],

        DATASET_RANGES[
            "Torque [Nm]"
        ][0],

        DATASET_RANGES[
            "Tool wear [min]"
        ][0]
    ],

    "Maximum": [
        "H",
        DATASET_RANGES[
            "Air temperature [K]"
        ][1],

        DATASET_RANGES[
            "Process temperature [K]"
        ][1],

        DATASET_RANGES[
            "Rotational speed [rpm]"
        ][1],

        DATASET_RANGES[
            "Torque [Nm]"
        ][1],

        DATASET_RANGES[
            "Tool wear [min]"
        ][1]
    ]

})

st.dataframe(
    range_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MAINTENANCE THRESHOLDS
# ============================================================

st.header(
    "🔧 Maintenance Thresholds"
)

maintenance_table = pd.DataFrame({

    "Parameter": [
        "Failure Probability",
        "Tool Wear",
        "Tool Wear",
        "Tool Wear",
        "Torque",
        "Temperature Difference",
        "Rotational Speed"
    ],

    "Condition": [
        "≥ 75%",
        "≥ 200 min",
        "≥ 150 min",
        "≥ 100 min",
        "> 50 Nm",
        "> 10 K",
        "< 1200 rpm"
    ],

    "Action": [
        "Immediate maintenance",
        "Immediate maintenance",
        "Maintenance required soon",
        "Schedule maintenance",
        "Sensor warning",
        "Sensor warning",
        "Sensor warning"
    ]
})

st.dataframe(
    maintenance_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FAILURE TYPE REFERENCE
# ============================================================

st.header(
    "📚 Failure Type Reference"
)

failure_reference = pd.DataFrame({

    "Code": [
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ],

    "Meaning": [
        "Tool Wear Failure",
        "Heat Dissipation Failure",
        "Power Failure",
        "Overstrain Failure",
        "Random Failure"
    ]

})

st.dataframe(
    failure_reference,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.header(
    "ℹ️ Project Information"
)

st.write(
    """
    Project Title: Predictive Maintenance Using Machine Learning

    Dataset: AI4I 2020 Predictive Maintenance Dataset

    Machine Learning Models:
    • Logistic Regression
    • Random Forest
    • Support Vector Machine (SVM)

    Input Parameters:
    • Machine Type
    • Air Temperature
    • Process Temperature
    • Rotational Speed
    • Torque
    • Tool Wear

    Output:
    • Machine Failure Prediction
    • Failure Probability
    • Maintenance Recommendation
    • Estimated Maintenance Time
    • Sensor Warnings
    """
)

st.info(
    """
    Note: The maintenance time values such as 0, 10, 25 and
    50 operating hours are rule-based estimates created for
    this project. The AI4I dataset does not contain actual
    Remaining Useful Life (RUL) or maintenance-history data.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Predictive Maintenance System | Machine Learning + Sensor Analytics"
)   