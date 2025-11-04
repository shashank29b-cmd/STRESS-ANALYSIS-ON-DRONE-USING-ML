import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import glob
import os

class StressPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast ML Stress Predictor - ANSYS FEA Replacement")
        self.root.geometry("900x800")
        self.root.configure(bg='#f0f0f0')
        
        # Load model and preprocessing objects
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load the trained model, scaler, and encoder"""
        try:
            # Find most recent model file
            model_files = glob.glob('stress_predictor_model_*.pkl')
            if not model_files:
                messagebox.showerror("Error", "No trained model found! Please run the training script first.")
                self.root.destroy()
                return
            
            model_file = max(model_files, key=os.path.getctime)
            model_data = joblib.load(model_file)
            
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.use_scaled = model_data['use_scaled']
            self.feature_columns = model_data['feature_columns']
            
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            
            self.materials = list(self.label_encoder.classes_)
            
            print(f"✓ Model loaded: {self.model_name}")
            print(f"✓ Model file: {model_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Fast ML Stress Predictor",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            title_frame,
            text=f"Powered by {self.model_name} | <0.1s prediction vs 10min FEA",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel - Inputs
        left_panel = tk.LabelFrame(
            main_container,
            text="Input Parameters",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=15,
            pady=15
        )
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Geometry Section
        self.create_section(left_panel, "Geometry Parameters", 0)
        self.length_var = self.create_input(left_panel, "Length (mm):", "500", 1)
        self.width_var = self.create_input(left_panel, "Width (mm):", "50", 2)
        self.height_var = self.create_input(left_panel, "Height (mm):", "25", 3)
        self.thickness_var = self.create_input(left_panel, "Thickness (mm):", "10", 4)
        
        # Material Section
        self.create_section(left_panel, "Material Properties", 5)
        self.material_var = tk.StringVar(value=self.materials[0])
        tk.Label(left_panel, text="Material:", font=('Arial', 10), bg='white').grid(
            row=6, column=0, sticky='w', pady=5
        )
        material_combo = ttk.Combobox(
            left_panel,
            textvariable=self.material_var,
            values=self.materials,
            state='readonly',
            width=25
        )
        material_combo.grid(row=6, column=1, sticky='ew', pady=5)
        material_combo.bind('<<ComboboxSelected>>', self.update_material_properties)
        
        self.youngs_var = tk.StringVar()
        self.poisson_var = tk.StringVar()
        self.yield_var = tk.StringVar()
        self.density_var = tk.StringVar()
        
        self.create_readonly_input(left_panel, "Young's Modulus (MPa):", self.youngs_var, 7)
        self.create_readonly_input(left_panel, "Poisson's Ratio:", self.poisson_var, 8)
        self.create_readonly_input(left_panel, "Yield Strength (MPa):", self.yield_var, 9)
        self.create_readonly_input(left_panel, "Density (kg/m³):", self.density_var, 10)
        
        # Loading Section
        self.create_section(left_panel, "Loading Conditions", 11)
        self.force_var = self.create_input(left_panel, "Force (N):", "1000", 12)
        self.moment_var = self.create_input(left_panel, "Moment (N-mm):", "5000", 13)
        self.pressure_var = self.create_input(left_panel, "Pressure (MPa):", "0", 14)
        self.torque_var = self.create_input(left_panel, "Torque (N-mm):", "0", 15)
        
        self.load_type_var = tk.IntVar(value=0)
        tk.Label(left_panel, text="Load Type:", font=('Arial', 10), bg='white').grid(
            row=16, column=0, sticky='w', pady=5
        )
        load_frame = tk.Frame(left_panel, bg='white')
        load_frame.grid(row=16, column=1, sticky='w', pady=5)
        tk.Radiobutton(load_frame, text="Tensile", variable=self.load_type_var, value=0, bg='white').pack(side=tk.LEFT)
        tk.Radiobutton(load_frame, text="Bending", variable=self.load_type_var, value=1, bg='white').pack(side=tk.LEFT)
        tk.Radiobutton(load_frame, text="Combined", variable=self.load_type_var, value=2, bg='white').pack(side=tk.LEFT)
        
        self.boundary_var = tk.IntVar(value=0)
        tk.Label(left_panel, text="Boundary Condition:", font=('Arial', 10), bg='white').grid(
            row=17, column=0, sticky='w', pady=5
        )
        bc_frame = tk.Frame(left_panel, bg='white')
        bc_frame.grid(row=17, column=1, sticky='w', pady=5)
        tk.Radiobutton(bc_frame, text="Fixed-Free", variable=self.boundary_var, value=0, bg='white').pack(side=tk.LEFT)
        tk.Radiobutton(bc_frame, text="Fixed-Fixed", variable=self.boundary_var, value=1, bg='white').pack(side=tk.LEFT)
        tk.Radiobutton(bc_frame, text="Pinned", variable=self.boundary_var, value=2, bg='white').pack(side=tk.LEFT)
        
        # Buttons
        button_frame = tk.Frame(left_panel, bg='white')
        button_frame.grid(row=18, column=0, columnspan=2, pady=20)
        
        predict_btn = tk.Button(
            button_frame,
            text="PREDICT STRESS",
            command=self.predict_stress,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="CLEAR",
            command=self.clear_inputs,
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = tk.Button(
            button_frame,
            text="SAVE RESULT",
            command=self.save_result,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Right Panel - Results
        right_panel = tk.LabelFrame(
            main_container,
            text="Prediction Results",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=15,
            pady=15
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results display
        self.results_text = tk.Text(
            right_panel,
            font=('Courier', 10),
            bg='#ecf0f1',
            fg='#2c3e50',
            padx=10,
            pady=10,
            wrap=tk.WORD,
            height=25
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize material properties
        self.update_material_properties()
        
        # Configure grid weights
        left_panel.columnconfigure(1, weight=1)
    
    def create_section(self, parent, title, row):
        """Create a section header"""
        label = tk.Label(
            parent,
            text=title,
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            padx=5,
            pady=5
        )
        label.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(10, 5))
    
    def create_input(self, parent, label_text, default_value, row):
        """Create an input field"""
        tk.Label(parent, text=label_text, font=('Arial', 10), bg='white').grid(
            row=row, column=0, sticky='w', pady=5
        )
        var = tk.StringVar(value=default_value)
        entry = tk.Entry(parent, textvariable=var, font=('Arial', 10), width=25)
        entry.grid(row=row, column=1, sticky='ew', pady=5)
        return var
    
    def create_readonly_input(self, parent, label_text, var, row):
        """Create a readonly input field"""
        tk.Label(parent, text=label_text, font=('Arial', 10), bg='white').grid(
            row=row, column=0, sticky='w', pady=5
        )
        entry = tk.Entry(parent, textvariable=var, font=('Arial', 10), width=25, state='readonly')
        entry.grid(row=row, column=1, sticky='ew', pady=5)
    
    def update_material_properties(self, event=None):
        """Update material properties based on selection"""
        material_props = {
            'Aluminum_6061': {'E': 69000, 'nu': 0.33, 'yield': 276, 'density': 2700},
            'Steel_AISI_1020': {'E': 200000, 'nu': 0.29, 'yield': 350, 'density': 7850},
            'Titanium_Ti6Al4V': {'E': 113800, 'nu': 0.342, 'yield': 880, 'density': 4430},
            'Copper_C11000': {'E': 117000, 'nu': 0.355, 'yield': 220, 'density': 8940},
            'Stainless_Steel_304': {'E': 193000, 'nu': 0.29, 'yield': 215, 'density': 8000}
        }
        
        material = self.material_var.get()
        props = material_props[material]
        
        self.youngs_var.set(str(props['E']))
        self.poisson_var.set(str(props['nu']))
        self.yield_var.set(str(props['yield']))
        self.density_var.set(str(props['density']))
    
    def predict_stress(self):
        """Make stress prediction"""
        try:
            # Get input values
            length = float(self.length_var.get())
            width = float(self.width_var.get())
            height = float(self.height_var.get())
            thickness = float(self.thickness_var.get())
            
            youngs = float(self.youngs_var.get())
            poisson = float(self.poisson_var.get())
            yield_strength = float(self.yield_var.get())
            density = float(self.density_var.get())
            
            force = float(self.force_var.get())
            moment = float(self.moment_var.get())
            pressure = float(self.pressure_var.get())
            torque = float(self.torque_var.get())
            
            load_type = self.load_type_var.get()
            boundary = self.boundary_var.get()
            
            material = self.material_var.get()
            material_encoded = self.label_encoder.transform([material])[0]
            
            # Calculate derived properties
            area = width * thickness
            I = (width * height**3) / 12
            
            # Create feature array
            features = np.array([[
                length, width, height, thickness, area, I,
                youngs, poisson, yield_strength, density,
                force, moment, pressure, torque,
                load_type, boundary, material_encoded
            ]])
            
            # Make prediction
            start_time = datetime.now()
            
            if self.use_scaled:
                features_scaled = self.scaler.transform(features)
                predicted_stress = self.model.predict(features_scaled)[0]
            else:
                predicted_stress = self.model.predict(features)[0]
            
            end_time = datetime.now()
            prediction_time = (end_time - start_time).total_seconds() * 1000
            
            # Calculate safety factor
            safety_factor = yield_strength / predicted_stress if predicted_stress > 0 else 100
            
            # Store result for saving
            self.last_result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'material': material,
                'length': length,
                'width': width,
                'height': height,
                'thickness': thickness,
                'force': force,
                'moment': moment,
                'pressure': pressure,
                'torque': torque,
                'predicted_stress': predicted_stress,
                'safety_factor': safety_factor,
                'prediction_time': prediction_time
            }
            
            # Display results
            self.display_results(predicted_stress, safety_factor, prediction_time, yield_strength)
            
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
    
    def display_results(self, stress, safety_factor, pred_time, yield_strength):
        """Display prediction results"""
        self.results_text.delete(1.0, tk.END)
        
        # Determine safety status
        if safety_factor > 2:
            status = "SAFE"
            status_color = "green"
        elif safety_factor > 1:
            status = "CAUTION"
            status_color = "orange"
        else:
            status = "UNSAFE"
            status_color = "red"
        
        result_text = f"""
{'='*50}
    STRESS PREDICTION RESULTS
{'='*50}

Material: {self.material_var.get().replace('_', ' ')}
Model: {self.model_name}

PREDICTED STRESS
{'─'*50}
Maximum Stress:     {stress:.2f} MPa
Yield Strength:     {yield_strength:.2f} MPa
Safety Factor:      {safety_factor:.2f}

STATUS: {status}
{'─'*50}

PERFORMANCE
{'─'*50}
Prediction Time:    {pred_time:.2f} ms
vs FEA Time:        ~10 minutes
Speed-up:           ~{(600000/pred_time):.0f}x faster

GEOMETRY
{'─'*50}
Length:             {self.length_var.get()} mm
Width:              {self.width_var.get()} mm
Height:             {self.height_var.get()} mm
Thickness:          {self.thickness_var.get()} mm

LOADING
{'─'*50}
Force:              {self.force_var.get()} N
Moment:             {self.moment_var.get()} N-mm
Pressure:           {self.pressure_var.get()} MPa
Torque:             {self.torque_var.get()} N-mm
Load Type:          {['Tensile', 'Bending', 'Combined'][self.load_type_var.get()]}
Boundary:           {['Fixed-Free', 'Fixed-Fixed', 'Pinned'][self.boundary_var.get()]}

{'='*50}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}
"""
        
        self.results_text.insert(1.0, result_text)
        
        # Color code the status
        if status == "SAFE":
            self.results_text.tag_add("status", "8.8", "8.12")
            self.results_text.tag_config("status", foreground="green", font=('Courier', 10, 'bold'))
        elif status == "CAUTION":
            self.results_text.tag_add("status", "8.8", "8.15")
            self.results_text.tag_config("status", foreground="orange", font=('Courier', 10, 'bold'))
        else:
            self.results_text.tag_add("status", "8.8", "8.14")
            self.results_text.tag_config("status", foreground="red", font=('Courier', 10, 'bold'))
    
    def clear_inputs(self):
        """Clear all input fields"""
        self.length_var.set("500")
        self.width_var.set("50")
        self.height_var.set("25")
        self.thickness_var.set("10")
        self.force_var.set("1000")
        self.moment_var.set("5000")
        self.pressure_var.set("0")
        self.torque_var.set("0")
        self.load_type_var.set(0)
        self.boundary_var.set(0)
        self.results_text.delete(1.0, tk.END)
    
    def save_result(self):
        """Save the last prediction result to CSV"""
        if not hasattr(self, 'last_result'):
            messagebox.showwarning("No Result", "Please make a prediction first before saving.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"stress_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if filename:
                df = pd.DataFrame([self.last_result])
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Result saved to:\n{filename}")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save result: {str(e)}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StressPredictorGUI(root)
    root.mainloop()
