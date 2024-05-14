import streamlit as st
import numpy as np
from organ_bioprint_env import OrganBioprintEnv
from rl_agent import BioprintingAgent
from stable_baselines3 import PPO

# Set page title and favicon
st.set_page_config(page_title="Bioprinting Simulator", page_icon=":microscope:")

# Define the Streamlit app
def main():
    # Create sidebar for user inputs
    st.sidebar.title("Bioprinting Parameters")
    resolution = st.sidebar.slider("Resolution", min_value=32, max_value=128, value=64, step=16)
    materials = st.sidebar.multiselect("Materials", ["Hydrogel", "Collagen", "Gelatin"], default=["Hydrogel"])
    num_steps = st.sidebar.number_input("Number of Steps", min_value=100, max_value=10000, value=1000, step=100)
    
    # Create the bioprinting environment
    env = OrganBioprintEnv(resolution=(resolution, resolution, resolution), materials=materials)
    
    # Load or train the reinforcement learning agent
    if st.sidebar.button("Train Agent"):
        hyperparameters = {
            "learning_rate": 3e-4,
            "lr_decay_steps": 1000,
            "lr_decay_rate": 0.9,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "tensorboard_log": "./tensorboard_logs/",
            "total_timesteps": num_steps
        }
        model = PPO(policy=BioprintingAgent, env=env, **hyperparameters)
        model.learn(total_timesteps=num_steps)
    else:
        try:
            model = PPO.load("trained_bioprinting_agent")
        except FileNotFoundError:
            st.warning("No trained agent found. Please train the agent first.")
            return
    
    # Simulate the bioprinting process
    if st.button("Start Bioprinting"):
        observation = env.reset()
        for _ in range(num_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, _ = env.step(action)
            if done:
                break
        
        # Display the bioprinted structure
        bioprinted_structure = env.render(mode="rgb_array")
        st.image(bioprinted_structure, caption="Bioprinted Structure", use_column_width=True)
    
    # Provide educational content
    st.header("Educational Content")
    st.markdown(
        """
        Bioprinting is an emerging technology that combines 3D printing with biomaterials to fabricate complex tissue structures. It has the potential to revolutionize regenerative medicine and drug testing.
        
        The bioprinting process involves the following steps:
        1. Design: Create a 3D model of the desired tissue structure using computer-aided design (CAD) software.
        2. Material Selection: Choose appropriate biomaterials, such as hydrogels, that can support cell growth and mimic the extracellular matrix.
        3. Printing: Use a bioprinter to dispense the biomaterials and cells layer by layer, following the designed 3D model.
        4. Incubation: Place the bioprinted structure in an incubator to allow the cells to proliferate and form the desired tissue.
        
        Bioprinting has numerous applications, including:
        - Tissue Engineering: Bioprinting can be used to create functional tissues and organs for transplantation or research purposes.
        - Drug Testing: Bioprinted tissue models can serve as a platform for testing the efficacy and toxicity of new drugs.
        - Personalized Medicine: Bioprinting enables the creation of patient-specific tissue constructs based on individual medical data.
        
        However, bioprinting also faces several challenges, such as:
        - Vascularization: Creating a functional vascular network within the bioprinted tissue remains a significant hurdle.
        - Material Limitations: Developing biomaterials that can accurately mimic the complex properties of native tissues is an ongoing challenge.
        - Scalability: Scaling up the bioprinting process to produce larger, clinically relevant tissue constructs requires further advancements.
        
        Despite these challenges, bioprinting holds immense potential for advancing regenerative medicine and improving patient outcomes in the future.
        """
    )

if __name__ == "__main__":
    main()