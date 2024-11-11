import streamlit as st

st.set_page_config(
    page_title="Speak With Data",
    page_icon="🎙️",
    initial_sidebar_state="expanded",
)

Info = st.Page("ui/about.py", title="About", icon=":material/info:", default=True)

Dashboard = st.Page("ui/dashboard.py", title="Dashboard", icon=":material/dashboard:")

ModelConfig = st.Page("ui/settings.py", title="Settings", icon=":material/settings:")

pg = st.navigation(
    {
        "Home": [Info],
        "Dashboard": [Dashboard],
        "Settings": [ModelConfig],
    }
)
st.sidebar.markdown("Made with ❤️ by [LordPotter](https://github.com/RezaS0)")
pg.run()
