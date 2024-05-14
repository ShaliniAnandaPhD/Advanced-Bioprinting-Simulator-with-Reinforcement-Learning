import plotly.graph_objects as go
import numpy as np

def visualize_bioprinted_structure(bioprinted_structure):
    # Get the dimensions of the bioprinted structure
    nx, ny, nz, num_materials = bioprinted_structure.shape
    
    # Create a list to store the traces for each material
    traces = []
    
    # Define the color map for each material
    color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create a trace for each material
    for material_index in range(num_materials):
        # Get the indices where the material is present
        material_indices = np.where(bioprinted_structure[:, :, :, material_index] > 0.5)
        
        # Create a scatter3d trace for the material
        trace = go.Scatter3d(
            x=material_indices[0],
            y=material_indices[1],
            z=material_indices[2],
            mode='markers',
            marker=dict(
                size=3,
                color=color_map[material_index % len(color_map)],
                opacity=0.8
            ),
            name=f"Material {material_index + 1}"
        )
        
        traces.append(trace)
    
    # Create the layout for the plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X', range=[0, nx]),
            yaxis=dict(title='Y', range=[0, ny]),
            zaxis=dict(title='Z', range=[0, nz])
        ),
        title="Bioprinted Structure Visualization",
        width=800,
        height=600
    )
    
    # Create the figure and add the traces and layout
    fig = go.Figure(data=traces, layout=layout)
    
    return fig

def visualize_bioprinting_process(bioprinted_structures):
    # Create a list to store the frames
    frames = []
    
    # Create a frame for each step in the bioprinting process
    for structure in bioprinted_structures:
        frame = visualize_bioprinted_structure(structure)
        frames.append(frame)
    
    # Create the layout for the animation
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X', range=[0, structure.shape[0]]),
            yaxis=dict(title='Y', range=[0, structure.shape[1]]),
            zaxis=dict(title='Z', range=[0, structure.shape[2]])
        ),
        title="Bioprinting Process Animation",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None])]
        )],
        width=800,
        height=600
    )
    
    # Create the figure and add the frames and layout
    fig = go.Figure(data=frames[0].data, layout=layout, frames=frames)
    
    return fig