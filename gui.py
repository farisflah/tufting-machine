import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label, Entry, Button, StringVar, BooleanVar, Listbox, Scrollbar, MULTIPLE, END
from PIL import Image, ImageTk, ImageDraw
import os
import cv2
import config
from img_processor import extract_shapes
from vector_creator import circumference_vectors, filling_vectors

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Separator")

        # File selection
        self.image_path = None
        self.img_label = tk.Label(root, text="No Image Selected")
        self.img_label.pack()

        self.select_btn = tk.Button(root, text="Select Image", command=self.load_image)
        self.select_btn.pack()

        # Process button
        self.process_btn = tk.Button(root, text="Separate Colors", command=self.process_image, state=tk.DISABLED)
        self.process_btn.pack()

        # "Create Vectors" button (for future implementation)
        self.vector_btn = tk.Button(root, text="Create Vectors", command=self.create_vectors)
        self.vector_btn.pack()

        # Status message
        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

        # Directory where separated colors will be saved
        self.output_folder = "output/"
        os.makedirs(self.output_folder, exist_ok=True)

        # Canvas to show selected image
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        # Settings button (top-right corner)
        img_pil = Image.open("settings_icon.png").resize((24, 24))  # Resize icon to fit
        settings_icon = ImageTk.PhotoImage(img_pil)
        self.settings_icon = settings_icon  # Keep reference
        self.settings_btn = tk.Button(root, image=settings_icon, command=self.open_settings, borderwidth=0)
        self.settings_btn.place(relx=1.0, x=-10, y=10, anchor='ne')

    def load_image(self):
        """ Open file dialog and select an image """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.bmp;*.tiff;*.webp;*.avif")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.img_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.process_btn.config(state=tk.NORMAL)  # Enable processing button
            self.vector_btn.config(state=tk.NORMAL)  # Disable vectors button until processing is done

    def display_image(self, path):
        """ Display the selected image on the GUI """
        try:
            # Try loading with OpenCV first (supports more formats)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("OpenCV could not read the image.")

            # Convert image to RGB (OpenCV loads images in BGR)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                # Grayscale image (2D array)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Convert OpenCV image (numpy array) to PIL Image
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((400, 400))  # Resize for display

        except Exception as e:
            print(f"OpenCV failed to load the image. Error: {e}")
            try:
                # Fallback to PIL if OpenCV fails
                img_pil = Image.open(path)
                img_pil = img_pil.resize((400, 400))  # Resize for display
            except Exception as pil_error:
                print(f"PIL failed to load the image. Error: {pil_error}")
                messagebox.showerror("Error", "Cannot open the selected image file.")
                return

        # Convert PIL image to Tkinter format
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(200, 200, image=img_tk)
        self.canvas.image = img_tk  # Keep reference

    def process_image(self):
        """ Process the image and separate colors and shapes """
        if not self.image_path:
            messagebox.showerror("Error", "No image selected!")
            return

        self.status_label.config(text="Processing image... Please wait.")
        self.root.update_idletasks()  # Force update the GUI

        # Separate colors and shapes
        separated_images = extract_shapes(self.image_path, self.output_folder)

        if separated_images:
            messagebox.showinfo("Success",
                                f"Separated {len(separated_images)} shapes and saved to {self.output_folder}")
        else:
            messagebox.showerror("Error", "No shapes were detected! Try selecting another image.")

        self.status_label.config(text="Processing complete.")

    def open_settings(self):
        """Open a settings window to update config values."""
        settings_win = Toplevel(self.root)
        settings_win.title("Settings")

        Label(settings_win, text="Target Vector Length (mm):").grid(row=0, column=0, padx=10, pady=5)
        length_entry = Entry(settings_win)
        length_entry.insert(0, str(config.TARGET_VECTOR_LENGTH))
        length_entry.grid(row=0, column=1)

        Label(settings_win, text="Target Vector Width (mm):").grid(row=1, column=0, padx=10, pady=5)
        width_entry = Entry(settings_win)
        width_entry.insert(0, str(config.TARGET_VECTOR_WIDTH))
        width_entry.grid(row=1, column=1)

        Label(settings_win, text="Threshold Value (0-255):").grid(row=2, column=0, padx=10, pady=5)#, sticky='e')
        threshold_entry = Entry(settings_win)
        threshold_entry.insert(0, str(config.THRESHOLD_VALUE))
        threshold_entry.grid(row=2, column=1)

        Label(settings_win, text="Number of Clusters:").grid(row=3, column=0, padx=10, pady=5)#, sticky='e')
        cluster_entry = Entry(settings_win)
        cluster_entry.insert(0, str(config.NUM_CLUSTERS))
        cluster_entry.grid(row=3, column=1)

        Label(settings_win, text="Number of Epsilon:").grid(row=4, column=0, padx=10, pady=5)  # , sticky='e')
        epsilon_entry = Entry(settings_win)
        epsilon_entry.insert(0, str(config.EPSILON))
        epsilon_entry.grid(row=4, column=1)

        Label(settings_win, text="Number of color distance threshold:").grid(row=5, column=0, padx=10, pady=5)  # , sticky='e')
        color_distance_threshold_entry = Entry(settings_win)
        color_distance_threshold_entry.insert(0, str(config.COLOR_DISTANCE_THRESHOLD))
        color_distance_threshold_entry.grid(row=5, column=1)

        Label(settings_win, text="Number of minimal percentage:").grid(row=6, column=0, padx=10, pady=5)  # , sticky='e')
        min_percentage_entry = Entry(settings_win)
        min_percentage_entry.insert(0, str(config.MIN_PERCENTAGE))
        min_percentage_entry.grid(row=6, column=1)

        def save_config():
            try:
                config.TARGET_VECTOR_LENGTH = float(length_entry.get())
                config.TARGET_VECTOR_WIDTH = float(width_entry.get())
                config.NUM_CLUSTERS = int(threshold_entry.get())
                config.THRESHOLD_VALUE = int(cluster_entry.get())
                config.EPSILON = float(epsilon_entry.get())
                config.COLOR_DISTANCE_THRESHOLD = float(color_distance_threshold_entry.get())
                config.MIN_PERCENTAGE = int(min_percentage_entry.get())
                messagebox.showinfo("Success", "Settings updated successfully.")
                settings_win.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers.")

        Button(settings_win, text="Save", command=save_config).grid(row=7, column=0, columnspan=2, pady=10)

    def create_vectors(self):
        """GUI workflow to generate vectors from selected shapes with overlay preview."""

        window = Toplevel(self.root)
        window.title("Vector Generation Wizard")

        # 1. List all colors
        images = [f for f in os.listdir(self.output_folder) if f.lower().endswith(('.png', '.jpg'))]
        colors = {}
        for fname in images:
            if fname.startswith("color_") and "_shape_" in fname and not fname.endswith(
                    ("_circumference.png", "_filling.png")):
                parts = fname.split("_")
                color_id = parts[1]
                colors.setdefault(color_id, []).append(fname)

        if not colors:
            messagebox.showerror("Error", "No shape images found.")
            return

        selected_color = StringVar()
        selected_color.set(next(iter(colors)))  # Default to first color

        Label(window, text="Select Color ID:").pack()
        color_dropdown = tk.OptionMenu(window, selected_color, *colors.keys())
        color_dropdown.pack()

        preview_canvas = tk.Canvas(window, width=400, height=400)
        preview_canvas.pack()

        shape_label = Label(window, text="Select Shape(s):")
        shape_label.pack()

        shape_listbox = Listbox(window, selectmode=MULTIPLE, width=40, exportselection=False)
        shape_listbox.pack()

        scrollbar = Scrollbar(window)
        scrollbar.pack(side="right", fill="y")
        shape_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=shape_listbox.yview)

        # Cache original image and update preview
        self.base_image = None

        def render_preview():
            """Merge base color image with all selected shape filling masks."""
            selected_shapes = [shape_listbox.get(i) for i in shape_listbox.curselection()]
            if self.base_image is None:
                return

            preview_img = self.base_image.copy()

            for shape_file in selected_shapes:
                base_name = os.path.splitext(shape_file)[0]
                fill_path = os.path.join(self.output_folder, base_name + "_filling.png")

                if os.path.exists(fill_path):
                    fill_img = Image.open(fill_path).convert("RGBA").resize(preview_img.size)
                    shape_mask = Image.new("RGBA", preview_img.size, (0, 0, 0, 0)) # Create a black mask where filling
                    fill_data = fill_img.getdata()
                    black_overlay = [(0, 0, 0, 120) if px[3] > 0 else (0, 0, 0, 0) for px in fill_data]
                    shape_mask.putdata(black_overlay)
                    preview_img = Image.alpha_composite(preview_img, shape_mask)

            preview_tk = ImageTk.PhotoImage(preview_img)
            preview_canvas.create_image(200, 200, image=preview_tk)
            preview_canvas.image = preview_tk

        def update_shapes(*_):
            shape_listbox.delete(0, END)
            selected = selected_color.get()
            if selected in colors:
                for shape in sorted(colors[selected]):
                    shape_listbox.insert(END, shape)

            # Load and store base image
            preview_path = os.path.join(self.output_folder, f"color_{selected}.png")
            if os.path.exists(preview_path):
                try:
                    base_img = Image.open(preview_path).convert("RGBA").resize((400, 400))
                    self.base_image = base_img
                except Exception as e:
                    print("Failed preview:", e)

            render_preview()

        # Update overlay whenever selection changes
        def on_shape_select(event):
            render_preview()

        shape_listbox.bind('<<ListboxSelect>>', on_shape_select)

        selected_color.trace_add("write", update_shapes)
        update_shapes()

        # Vector type checkboxes
        circ_var = BooleanVar(value=True)
        fill_var = BooleanVar(value=False)

        Label(window, text="Vector Types:").pack()
        tk.Checkbutton(window, text="Circumference Vectors", variable=circ_var).pack()
        tk.Checkbutton(window, text="Filling Vectors", variable=fill_var).pack()

        Label(window, text="Real Width (mm):").pack()
        real_width_entry = Entry(window)
        real_width_entry.pack()

        def run_generation():
            shape_indexes = shape_listbox.curselection()
            if not shape_indexes:
                messagebox.showerror("Error", "No shapes selected.")
                return

            try:
                real_width = float(real_width_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for real width.")
                return

            shapes = [shape_listbox.get(i) for i in shape_indexes]
            for shape_file in shapes:
                base_path = os.path.splitext(os.path.join(self.output_folder, shape_file))[0]
                if circ_var.get():
                    circ_path = base_path + "_circumference.png"
                    if os.path.exists(circ_path):
                        output_txt = base_path + "_circumference_vectors.txt"
                        print(f"Generating circumference vectors for {circ_path}")
                        circumference_vectors(circ_path, output_txt, real_width)
                    else:
                        print(f"Missing: {circ_path}")

                if fill_var.get():
                    fill_path = base_path + "_filling.png"
                    if os.path.exists(fill_path):
                        output_txt = base_path + "_filling_vectors.txt"
                        print(f"Generating filling vectors for {fill_path}")
                        filling_vectors(fill_path, output_txt, real_width)
                    else:
                        print(f"Missing: {fill_path}")

            messagebox.showinfo("Done", "Vector generation complete!")

        Button(window, text="Generate Vectors", command=run_generation).pack(pady=10)


def run_gui():
    """ Run the GUI from main.py """
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
