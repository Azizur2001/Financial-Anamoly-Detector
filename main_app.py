import streamlit as st
from groq import Groq
import os
import pickle
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


def load_knn_model():
    """Load the pre-trained KNN model."""
    try:
        with open("knn_model_tuned.pkl", "rb") as f:
            knn_model = pickle.load(f)
        st.write("Model loaded successfully.")
        return knn_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

def load_backtesting_data():
    """Load the backtesting dataset."""
    try:
        df = pd.read_csv("backtesting_data_with_predictions.csv")
        st.write("Backtesting data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading backtesting data: {e}")
        raise

def detect_anomalies(data, model, feature_names, threshold=0.5):
    """Detect anomalies in the given dataset."""
    try:
        # Align data columns with training features
        st.write("Aligning feature columns...")
        numeric_features = feature_names  # Ensure numeric-only columns
        data = data[numeric_features]
        st.write("Feature columns aligned successfully.")

        # Ensure model supports predict_proba
        if hasattr(model, "predict_proba"):
            st.write("Generating anomaly probabilities...")
            anomaly_probs = model.predict_proba(data)[:, 1]
        else:
            raise ValueError("The model does not support probability predictions.")

        anomaly_classes = (anomaly_probs >= threshold).astype(int)

        # Add predictions to the dataset
        data["Anomaly_Probability"] = anomaly_probs
        data["Anomaly_Flag"] = anomaly_classes

        # Add recommendations
        data["Recommendation"] = data["Anomaly_Flag"].apply(lambda x: "Sell" if x == 1 else "Hold")

        st.write("Anomaly detection completed.")
        return data
    except Exception as e:
        st.error(f"An error occurred during anomaly detection: {e}")
        raise


def main():
    st.title("Anomaly Detection with Groq-Powered Chatbot")

    # Initialize Groq client
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Load the model
    knn_model = load_knn_model()

    # Debug: Check the model type
    st.write(f"Loaded model type: {type(knn_model)}")

    # Load the backtesting data
    backtesting_data = load_backtesting_data()

    # Extract feature names from the model (if available)
    if hasattr(knn_model, "n_features_in_"):
        numeric_features = backtesting_data.select_dtypes(include=["float64", "int64"]).columns
        feature_names = numeric_features[:knn_model.n_features_in_]
        st.write("Extracted numeric feature names:", feature_names)
    else:
        st.error("The model does not have attribute 'n_features_in_' to verify features.")
        raise AttributeError("Missing 'n_features_in_' attribute in the model.")

    # Run anomaly detection only if not already done
    if "anomaly_results" not in st.session_state:
        if st.button("Run Anomaly Detection"):
            try:
                # Run anomaly detection
                st.session_state.anomaly_results = detect_anomalies(backtesting_data, knn_model, feature_names)
                st.session_state.results_saved = False  # Flag to track if results are saved
            except Exception as e:
                st.error(f"An error occurred during anomaly detection: {e}")
    else:
        st.write("Anomaly detection already completed.")

    # Display results if available
    if "anomaly_results" in st.session_state:
        # Allow dynamic row selection using a slider
        max_rows = len(st.session_state.anomaly_results)
        num_rows = st.slider("Select number of rows to display:", min_value=5, max_value=max_rows, value=10)

        # Display the selected rows
        st.dataframe(st.session_state.anomaly_results.head(num_rows), use_container_width=True)

        # Save results to CSV (optional)
        if not st.session_state.get("results_saved", False):
            st.session_state.anomaly_results.to_csv("anomaly_detection_results.csv", index=False)
            st.session_state.results_saved = True
            st.success("Results saved to 'anomaly_detection_results.csv'")

    # Chatbot section
    st.header("Groq-Powered Chatbot")

    # Initialize session state for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from the session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input section
    if prompt := st.chat_input("Type your message:"):
        # Add the user's message to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the user input with Groq
        with st.chat_message("assistant"):
            # Placeholder for response generation
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Call Groq API to get a chat completion
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    model="llama-3.3-70b-versatile",
                )

                # Extract the assistant's response
                assistant_message = chat_completion.choices[0].message.content
                full_response += assistant_message

                # Display the assistant's response
                message_placeholder.markdown(full_response)

                # Add the assistant's response to the session state
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"An error occurred during chat processing: {e}")

if __name__ == "__main__":
    main()
