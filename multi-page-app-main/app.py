import streamlit as st
from multiapp import MultiApp
from apps import home, prediction # import your app modules here

app = MultiApp()



# Add all your application here
app.add_app("Data Analysis and Data Processing", home.app)
app.add_app("Predicition", prediction.app)

# The main app
app.run()
