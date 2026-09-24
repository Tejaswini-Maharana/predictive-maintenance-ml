
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go



# PAGE CONFIG

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    .status-card {
        padding: 20px;
        border-radius: 12px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# PATHS

MODEL_PATHS = [
    "models/predictive_maintenance_model.pkl",
    "model/predictive_maintenance_model.pkl"
]

DATASET_PATH = "ai4i2020.csv"


# LOAD MODEL

@st.cache_resource
def load_model():

    for path in MODEL_PATHS:

        if os.path.exists(path):

            with open(path, "rb") as file:
                return pickle.load(file)

    return None


# LOAD DATA

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_PATH):
        return None

    return pd.read_csv(DATASET_PATH)


model = load_model()
df = load_dataset()


# ERROR HANDLING

if model is None:

    st.error(
        "Trained model not found."
    )

    st.info(
        "Please train your model first and make sure "
        "predictive_maintenance_model.pkl exists inside "
        "the models/ folder."
    )

    st.stop()


if df is None:

    st.error(
        "ai4i2020.csv was not found."
    )

    st.stop()


# CONSTANTS

TYPE_MAPPING = {
    "L": 0,
    "M": 1,
    "H": 2
}


SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


FAILURE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


FAILURE_NAMES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure"
}


# MAINTENANCE FUNCTION

def maintenance_recommendation(
    probability,
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


    if probability >= 0.75 or tool_wear >= 200:

        status = "IMMEDIATE MAINTENANCE REQUIRED"
        estimated_hours = 0

    elif probability >= 0.50 or tool_wear >= 150:

        status = "MAINTENANCE REQUIRED SOON"
        estimated_hours = 10

    elif probability >= 0.25 or tool_wear >= 100:

        status = "MAINTENANCE SHOULD BE SCHEDULED"
        estimated_hours = 25

    else:

        status = "MACHINE CONDITION NORMAL"
        estimated_hours = 50


    return status, estimated_hours, warnings


# SIDEBAR

with st.sidebar:

    st.title("⚙️ Predictive Maintenance")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 Machine Prediction",
            "📊 Data Analytics",
            "⚠️ Failure Analysis",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.caption(
        "Machine Learning + Sensor Analytics"
    )


# HEADER

st.markdown(
    '<div class="dashboard-title">'
    '⚙️ Predictive Maintenance Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Machine failure prediction and maintenance monitoring system'
    '</div>',
    unsafe_allow_html=True
)


# DASHBOARD PAGE

if page == "🏠 Dashboard":

    st.subheader("📌 System Overview")

    total_machines = len(df)

    total_failures = int(
        df["Machine failure"].sum()
    )

    failure_rate = (
        total_failures / total_machines
    ) * 100

    avg_tool_wear = df[
        "Tool wear [min]"
    ].mean()

    avg_torque = df[
        "Torque [Nm]"
    ].mean()


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Machines",
            f"{total_machines:,}"
        )


    with col2:

        st.metric(
            "Recorded Failures",
            f"{total_failures:,}"
        )


    with col3:

        st.metric(
            "Failure Rate",
            f"{failure_rate:.2f}%"
        )


    with col4:

        st.metric(
            "Average Tool Wear",
            f"{avg_tool_wear:.1f} min"
        )


    st.markdown("---")


    # FAILURE DISTRIBUTION

    col1, col2 = st.columns(2)


    with col1:

        failure_counts = pd.DataFrame({
            "Condition": [
                "Normal",
                "Failure"
            ],
            "Machines": [
                int((df["Machine failure"] == 0).sum()),
                int((df["Machine failure"] == 1).sum())
            ]
        })


        fig = px.bar(
            failure_counts,
            x="Condition",
            y="Machines",
            title="Machine Failure Distribution",
            text="Machines"
        )

        fig.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # MACHINE TYPES

    with col2:

        type_counts = (
            df["Type"]
            .value_counts()
            .reset_index()
        )

        type_counts.columns = [
            "Type",
            "Machines"
        ]

        type_counts["Type"] = (
            type_counts["Type"]
            .map({
                0: "L",
                1: "M",
                2: "H",
                "L": "L",
                "M": "M",
                "H": "H"
            })
        )


        fig = px.pie(
            type_counts,
            names="Type",
            values="Machines",
            title="Machine Type Distribution",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



    # SENSOR OVERVIEW

    st.subheader("📈 Sensor Overview")


    sensor_means = pd.DataFrame({
        "Sensor": [
            "Air Temperature",
            "Process Temperature",
            "Rotational Speed",
            "Torque",
            "Tool Wear"
        ],

        "Average": [
            df["Air temperature [K]"].mean(),
            df["Process temperature [K]"].mean(),
            df["Rotational speed [rpm]"].mean(),
            df["Torque [Nm]"].mean(),
            df["Tool wear [min]"].mean()
        ]
    })


    fig = px.bar(
        sensor_means,
        x="Sensor",
        y="Average",
        title="Average Sensor Values"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# MACHINE PREDICTION PAGE

elif page == "🔮 Machine Prediction":

    st.subheader(
        "🔮 Predict Machine Condition"
    )

    st.write(
        "Enter the current machine sensor readings."
    )



    # INPUTS

    col1, col2, col3 = st.columns(3)


    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )


    with col2:

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=float(
                df["Air temperature [K]"].min()
            ),
            max_value=float(
                df["Air temperature [K]"].max()
            ),
            value=float(
                df["Air temperature [K]"].mean()
            ),
            step=0.1
        )


    with col3:

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=float(
                df["Process temperature [K]"].min()
            ),
            max_value=float(
                df["Process temperature [K]"].max()
            ),
            value=float(
                df["Process temperature [K]"].mean()
            ),
            step=0.1
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=float(
                df["Rotational speed [rpm]"].min()
            ),
            max_value=float(
                df["Rotational speed [rpm]"].max()
            ),
            value=float(
                df["Rotational speed [rpm]"].mean()
            ),
            step=1.0
        )


    with col2:

        torque = st.number_input(
            "Torque [Nm]",
            min_value=float(
                df["Torque [Nm]"].min()
            ),
            max_value=float(
                df["Torque [Nm]"].max()
            ),
            value=float(
                df["Torque [Nm]"].mean()
            ),
            step=0.1
        )


    with col3:

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=float(
                df["Tool wear [min]"].min()
            ),
            max_value=float(
                df["Tool wear [min]"].max()
            ),
            value=float(
                df["Tool wear [min]"].mean()
            ),
            step=1.0
        )


    st.markdown("---")


    predict = st.button(
        "🔍 Predict Machine Condition",
        type="primary",
        use_container_width=True
    )


    if predict:

        input_data = pd.DataFrame(
            [[
                TYPE_MAPPING[machine_type],
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


        prediction = model.predict(
            input_data
        )[0]


        probability = model.predict_proba(
            input_data
        )[0][1]


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


        # RESULT CARDS
        st.markdown("---")

        st.subheader(
            "📊 Prediction Result"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            if prediction == 1:

                st.error(
                    "⚠️ MACHINE FAILURE DETECTED"
                )

            else:

                st.success(
                    "✅ MACHINE CONDITION NORMAL"
                )


        with col2:

            st.metric(
                "Failure Probability",
                f"{probability * 100:.2f}%"
            )


        with col3:

            if estimated_hours == 0:

                st.metric(
                    "Maintenance",
                    "IMMEDIATE"
                )

            else:

                st.metric(
                    "Estimated Maintenance",
                    f"{estimated_hours} hours"
                )


        # FAILURE PROBABILITY GAUGE

        st.subheader(
            "🎯 Failure Probability"
        )


        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={
                    "suffix": "%"
                },
                title={
                    "text": "Failure Risk"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "threshold": {
                        "line": {
                            "width": 4
                        },
                        "value": 75
                    }
                }
            )
        )


        gauge.update_layout(
            height=350
        )


        st.plotly_chart(
            gauge,
            use_container_width=True
        )


        # MAINTENANCE

        st.subheader(
            "🔧 Maintenance Assessment"
        )


        if status == "IMMEDIATE MAINTENANCE REQUIRED":

            st.error(status)

        elif status == "MAINTENANCE REQUIRED SOON":

            st.warning(status)

        elif status == "MAINTENANCE SHOULD BE SCHEDULED":

            st.info(status)

        else:

            st.success(status)


        # WARNINGS

        st.subheader(
            "⚠️ Sensor Warnings"
        )


        if warnings:

            for warning in warnings:

                st.warning(warning)

        else:

            st.success(
                "No major sensor warnings detected."
            )


        # SENSOR VALUES

        st.subheader(
            "📡 Current Sensor Values"
        )


        current_values = pd.DataFrame({
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
                f"{air_temperature:.2f} K",
                f"{process_temperature:.2f} K",
                f"{rotational_speed:.0f} rpm",
                f"{torque:.2f} Nm",
                f"{tool_wear:.0f} min"
            ]
        })


        st.dataframe(
            current_values,
            use_container_width=True,
            hide_index=True
        )


# DATA ANALYTICS PAGE

elif page == "📊 Data Analytics":

    st.subheader(
        "📊 Dataset Analytics"
    )


    # DATASET METRICS

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )


    with col4:

        st.metric(
            "Failure Cases",
            int(df["Machine failure"].sum())
        )


    st.markdown("---")


    # SENSOR DISTRIBUTIONS
    
    selected_sensor = st.selectbox(
        "Select Sensor",
        SENSOR_COLUMNS
    )


    fig = px.histogram(
        df,
        x=selected_sensor,
        color="Machine failure",
        nbins=40,
        title=f"{selected_sensor} Distribution"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # SENSOR VS FAILURE

    fig = px.box(
        df,
        x="Machine failure",
        y=selected_sensor,
        color="Machine failure",
        title=f"{selected_sensor} vs Machine Failure"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # CORRELATION

    st.subheader(
        "🔥 Sensor Correlation"
    )


    correlation = df[
        SENSOR_COLUMNS + ["Machine failure"]
    ].corr()


    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# FAILURE ANALYSIS

elif page == "⚠️ Failure Analysis":

    st.subheader(
        "⚠️ Failure Analysis"
    )


    failure_data = pd.DataFrame({
        "Failure Code": FAILURE_COLUMNS,

        "Failure Type": [
            FAILURE_NAMES[x]
            for x in FAILURE_COLUMNS
        ],

        "Cases": [
            int(df[x].sum())
            for x in FAILURE_COLUMNS
        ]
    })


    col1, col2 = st.columns(2)


    with col1:

        fig = px.bar(
            failure_data,
            x="Failure Code",
            y="Cases",
            text="Cases",
            title="Failure Type Frequency"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        fig = px.pie(
            failure_data,
            names="Failure Type",
            values="Cases",
            title="Failure Type Distribution"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.dataframe(
        failure_data,
        use_container_width=True,
        hide_index=True
    )


    # FAILURE RATE BY MACHINE TYPE

    st.subheader(
        "Machine Type vs Failure"
    )


    machine_failure = (
        df.groupby("Type")["Machine failure"]
        .mean()
        .reset_index()
    )


    machine_failure.columns = [
        "Machine Type",
        "Failure Rate"
    ]


    machine_failure["Failure Rate"] *= 100


    machine_failure["Machine Type"] = (
        machine_failure["Machine Type"]
        .map({
            0: "L",
            1: "M",
            2: "H",
            "L": "L",
            "M": "M",
            "H": "H"
        })
    )


    fig = px.bar(
        machine_failure,
        x="Machine Type",
        y="Failure Rate",
        text_auto=".2f",
        title="Failure Rate by Machine Type"
    )


    fig.update_yaxes(
        title="Failure Rate (%)"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ABOUT PAGE

elif page == "ℹ️ About Project":

    st.subheader(
        "ℹ️ About the Project"
    )


    st.markdown(
        """
        ## Predictive Maintenance Using Machine Learning

        This project uses machine learning to predict whether
        an industrial machine is likely to experience failure.

        ### Dataset

        AI4I 2020 Predictive Maintenance Dataset.

        ### Machine Learning Models

        - Logistic Regression
        - Random Forest
        - Support Vector Machine (SVM)

        ### Sensor Parameters

        - Machine Type
        - Air Temperature
        - Process Temperature
        - Rotational Speed
        - Torque
        - Tool Wear

        ### Dashboard Outputs

        - Machine failure prediction
        - Failure probability
        - Maintenance status
        - Sensor warnings
        - Dataset analytics
        - Failure type analysis
        """
    )


    st.info(
        """
        Important:

        The maintenance time estimates used in this project
        are rule-based estimates. The AI4I 2020 dataset does
        not contain actual Remaining Useful Life (RUL) or
        maintenance-history data.
        """
    )

# FOOTER

st.markdown("---")

st.caption(
    "Predictive Maintenance System | "
    "Machine Learning + Streamlit"
)