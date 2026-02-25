import streamlit as st
import numpy as np
from scipy.stats import t

# ---------------------------
# T-Test Function
# ---------------------------
def ttest(data, mu0, alpha=0.05, alternative="two-sided"):
    data = np.array(data)
    n = len(data)

    xbar = np.mean(data)
    s = np.std(data, ddof=1)
    se = s / np.sqrt(n)

    t_cal = (xbar - mu0) / se
    df = n - 1

    if alternative == "two-sided":
        p_value = 2 * (1 - t.cdf(abs(t_cal), df))
        reject = p_value < alpha

    elif alternative == "greater":
        p_value = 1 - t.cdf(t_cal, df)
        reject = p_value < alpha

    elif alternative == "less":
        p_value = t.cdf(t_cal, df)
        reject = p_value < alpha

    return xbar, s, t_cal, df, p_value, reject


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📊 One Sample T-Test Calculator")

st.write("Enter sample values separated by commas")

data_input = st.text_area("Sample Data", "10, 12, 9, 11, 10, 13, 12")
mu0 = st.number_input("Null Hypothesis Mean (μ₀)", value=10.0)
alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)
alternative = st.selectbox(
    "Alternative Hypothesis",
    ["two-sided", "greater", "less"]
)

if st.button("Calculate"):

    try:
        data = [float(x.strip()) for x in data_input.split(",")]

        xbar, s, t_cal, df, p_value, reject = ttest(
            data, mu0, alpha, alternative
        )

        st.subheader("Results")

        st.write(f"Sample Mean (x̄): {xbar:.4f}")
        st.write(f"Sample Std Dev (s): {s:.4f}")
        st.write(f"T Statistic: {t_cal:.4f}")
        st.write(f"Degrees of Freedom: {df}")
        st.write(f"P-value: {p_value:.6f}")

        if reject:
            st.error("Decision: Reject H₀")
        else:
            st.success("Decision: Fail to Reject H₀")

    except:
        st.warning("Please enter valid numeric values separated by commas.")