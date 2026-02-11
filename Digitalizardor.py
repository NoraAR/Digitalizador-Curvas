import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import csv
from PIL import Image, ImageTk, ImageDraw
import pandas as pd
from tkinter import filedialog

# ============================================================
#  CLASE PRINCIPAL MODULARIZADA
# ============================================================

class DigitizerApp:
    def __init__(self, root):
        self.root = root
        self.root.state("zoomed")
        self.root.title("Digitalizador")

        self.setup_states()
        self.setup_layout()
        self.setup_events()


    # ============================================================
    #  ESTADOS INTERNOS
    # ============================================================
    def setup_states(self):
        self.points_auto = []
        self.points_manual = []
        self.points_all = []
        self.curve_color = None

        self.axis_click_stage = 0
        self.axis_colors = ["#FF0000", "#00AA00", "#0000FF", "#AA00AA"]
        self.calib = {"x1": None, "x2": None, "y1": None, "y2": None}

        self.internal_mode = None
        self.rect_start = None
        self.rect_patch = None


    # ============================================================
    #  DISEÑO DE LA INTERFAZ GRÁFICA
    # ============================================================
    def setup_layout(self):
        # FIGURA
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        # PANEL DERECHO
        self.panel = tk.Frame(self.root, padx=10, pady=10)
        self.panel.pack(side="right", fill="y")

        # -------------------------------------------
        # 1. SECCIÓN: CARGAR IMAGEN
        # -------------------------------------------
        frame_load = self.make_section("Imagen")
        self.make_button(frame_load, "Cargar imagen", self.load_image)

        # -------------------------------------------
        # 2. SECCIÓN: CALIBRACIÓN DE EJES
        # -------------------------------------------
        frame_axes = self.make_section("Calibración")

        tk.Label(frame_axes, text="Valores reales").pack()

        self.x1_real = self.make_entry(frame_axes, "X1 real")
        self.x2_real = self.make_entry(frame_axes, "X2 real")
        self.y1_real = self.make_entry(frame_axes, "Y1 real")
        self.y2_real = self.make_entry(frame_axes, "Y2 real")

        tk.Label(frame_axes, text="Acción").pack()

        self.action_var = tk.StringVar(value="Calibrar ejes")

        self.make_radio(frame_axes, "Calibrar ejes", "Calibrar ejes")
        self.make_radio(frame_axes, "Seleccionar color curva", "Seleccionar color")

        # -------------------------------------------
        # 3. SECCIÓN: PUNTOS MANUALES
        # -------------------------------------------
        frame_points = self.make_section("Edición manual")

        self.make_button(frame_points, "Añadir puntos", self.enable_add_mode)
        self.make_button(frame_points, "Borrar sección", self.enable_delete_rectangle)

        # -------------------------------------------
        # 4. SECCIÓN: AUTO-DETECCIÓN
        # -------------------------------------------
        frame_auto = self.make_section("Auto digitización")

        self.make_button(frame_auto, "Auto detectar", self.run_autodetect)

        # -------------------------------------------
        # 5. SECCIÓN: EXPORTACIÓN
        # -------------------------------------------
        frame_export = self.make_section("Exportar / Reiniciar")

        self.make_button(frame_export, "Exportar CSV", self.export_csv)
        self.make_button(frame_export, "Reiniciar", self.reset_all)

        # -------------------------------------------
        # 6. SECCIÓN: MIN/MAX
        # -------------------------------------------
        frame_info = self.make_section("Información")

        tk.Label(frame_info, text="Punto mínimo:").pack()
        self.label_min = tk.Label(frame_info, text="---")
        self.label_min.pack()

        tk.Label(frame_info, text="Punto máximo:").pack()
        self.label_max = tk.Label(frame_info, text="---")
        self.label_max.pack()

        # -------------------------------------------
        # LUPA POPUP
        # -------------------------------------------
        self.zoom_popup = tk.Toplevel(self.root)
        self.zoom_popup.overrideredirect(True)
        self.zoom_popup.attributes("-topmost", True)
        self.zoom_label = tk.Label(self.zoom_popup)
        self.zoom_label.pack()
        self.zoom_popup.withdraw()


    # ============================================================
    #  CREACIÓN DE SECCIONES Y WIDGETS
    # ============================================================

    def make_section(self, title):
        frame = tk.LabelFrame(self.panel, text=title, padx=5, pady=5)
        frame.pack(fill="x", pady=6)
        return frame

    def make_button(self, parent, text, cmd):
        btn = tk.Button(parent, text=text, width=20, height=1, command=cmd)
        btn.pack(pady=3)
        return btn

    def make_radio(self, parent, text, value):
        rb = tk.Radiobutton(parent, text=text,
                            variable=self.action_var, value=value)
        rb.pack(anchor="w", pady=2)
        return rb

    def make_entry(self, parent, label):
        tk.Label(parent, text=label).pack()
        e = tk.Entry(parent, width=10)
        e.insert(0, "0")
        e.pack()
        return e


    # ============================================================
    #  EVENTOS DE LA FIGURA
    # ============================================================

    def setup_events(self):
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)


    # ============================================================
    #  LÓGICA PRINCIPAL (MISMA QUE ANTES)
    # ============================================================

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        self.img = mpimg.imread(path)
        self.ax.clear()
        self.ax.imshow(self.img)
        self.fig.canvas.draw_idle()


    # ---------------------------------------------------------
    # MODOS

    def enable_add_mode(self):
        self.internal_mode = "add"
        print("Modo: añadir puntos")

    def enable_delete_rectangle(self):
        self.internal_mode = "delete_rect"
        print("Modo: borrar sección")


    # ---------------------------------------------------------
    # CLICK PRINCIPAL

    def on_click(self, event):
        if event.xdata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Modo rectángulo
        if self.internal_mode == "delete_rect":
            self.rect_start = (x, y)
            return

        # Modo añadir punto
        if self.internal_mode == "add":
            real = self.pixel_to_real(x, y)
            if real:
                art = self.ax.plot(x, y, "o", markersize=5, color="yellow")[0]
                self.points_manual.append(real)
                self.points_all.append((x, y, real, art))
                self.fig.canvas.draw_idle()
            return

        action = self.action_var.get()

        # -------- CALIBRAR --------
        if action == "Calibrar ejes":
            if not self.validate_axes_inputs():
                print("Valores inválidos.")
                return

            idx = self.axis_click_stage
            color = self.axis_colors[idx]
            self.ax.plot(x, y, "o", color=color)

            if idx == 0: self.calib["x1"] = (x, y)
            elif idx == 1: self.calib["x2"] = (x, y)
            elif idx == 2: self.calib["y1"] = (x, y)
            elif idx == 3: self.calib["y2"] = (x, y)

            self.axis_click_stage = (self.axis_click_stage + 1) % 4
            self.fig.canvas.draw_idle()
            return

        # -------- SELECCIONAR COLOR --------
        if action == "Seleccionar color":
            self.curve_color = self.img[y, x, :3]
            print("Color seleccionado:", self.curve_color)

            self.points_auto = []
            self.points_manual = []
            self.points_all = []
            self.internal_mode = None

            self.ax.clear()
            self.ax.imshow(self.img)
            self.fig.canvas.draw_idle()
            return


    # ---------------------------------------------------------
    # SOLTAR MOUSE (BORRAR SECCIÓN)

    def on_release(self, event):
        if self.internal_mode != "delete_rect":
            return
        if event.xdata is None:
            return

        x1, y1 = self.rect_start
        x2, y2 = int(event.xdata), int(event.ydata)

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        # Remover dibujo del rectángulo
        if self.rect_patch:
            self.rect_patch.remove()
            self.rect_patch = None

        remaining = []
        removed = 0

        for (px, py, real, art) in self.points_all:
            if xmin <= px <= xmax and ymin <= py <= ymax:
                art.remove()
                removed += 1
                if real in self.points_manual:
                    self.points_manual.remove(real)
                if real in self.points_auto:
                    self.points_auto.remove(real)
            else:
                remaining.append((px, py, real, art))

        self.points_all = remaining
        self.fig.canvas.draw_idle()

        print(f"{removed} puntos eliminados.")

        self.rect_start = None


    # ---------------------------------------------------------
    # MOUSE MOVE (RECT + LUPA)

    def on_motion(self, event):
        self.update_zoom(event)

        if self.internal_mode == "delete_rect" and self.rect_start and event.xdata:
            x0, y0 = self.rect_start
            x1, y1 = int(event.xdata), int(event.ydata)

            if self.rect_patch:
                self.rect_patch.remove()

            self.rect_patch = self.ax.add_patch(
                plt.Rectangle((min(x0, x1), min(y0, y1)),
                              abs(x1 - x0), abs(y1 - y0),
                              fill=False, edgecolor="red", linewidth=1.5)
            )

            self.fig.canvas.draw_idle()


    # ---------------------------------------------------------
    # LUPA EXACTA

    def update_zoom(self, event):
        if event.xdata is None or not hasattr(self, "img"):
            self.zoom_popup.withdraw()
            return

        x, y = int(event.xdata), int(event.ydata)
        h, w = self.img.shape[:2]

        patch_radius = 20
        zoom_factor = 10

        x1, x2 = max(0, x - patch_radius), min(w, x + patch_radius)
        y1, y2 = max(0, y - patch_radius), min(h, y + patch_radius)

        patch = self.img[y1:y2, x1:x2]
        if patch.size == 0:
            self.zoom_popup.withdraw()
            return

        pil_img = Image.fromarray((patch * 255).astype(np.uint8)).resize(
            (patch.shape[1] * zoom_factor,
             patch.shape[0] * zoom_factor),
            Image.NEAREST
        )

        draw = ImageDraw.Draw(pil_img)

        for xx in range(0, pil_img.width, zoom_factor):
            draw.line((xx, 0, xx, pil_img.height), fill="gray")
        for yy in range(0, pil_img.height, zoom_factor):
            draw.line((0, yy, pil_img.width, yy), fill="gray")

        cx = (x - x1) * zoom_factor
        cy = (y - y1) * zoom_factor

        draw.line((cx, 0, cx, pil_img.height), fill="red")
        draw.line((0, cy, pil_img.width, cy), fill="red")

        draw.rectangle(
            (cx - zoom_factor//2, cy - zoom_factor//2,
             cx + zoom_factor//2, cy + zoom_factor//2),
            outline="yellow", width=2
        )

        tk_img = ImageTk.PhotoImage(pil_img)
        self.zoom_label.config(image=tk_img)
        self.zoom_label.image = tk_img

        self.zoom_popup.deiconify()
        self.zoom_popup.geometry(
            f"+{self.root.winfo_pointerx() + 20}"
            f"+{self.root.winfo_pointery() + 20}"
        )


    # ---------------------------------------------------------
    # UTILIDADES

    def validate_axes_inputs(self):
        try:
            float(self.x1_real.get())
            float(self.x2_real.get())
            float(self.y1_real.get())
            float(self.y2_real.get())
            return True
        except:
            return False


    def pixel_to_real(self, px, py):
        if None in self.calib.values():
            return None

        (x1, y1) = self.calib["x1"]
        (x2, y2) = self.calib["x2"]
        (y1p) = self.calib["y1"]
        (y2p) = self.calib["y2"]

        x1r = float(self.x1_real.get())
        x2r = float(self.x2_real.get())
        y1r = float(self.y1_real.get())
        y2r = float(self.y2_real.get())

        ax = (x2r - x1r) / (x2 - x1)
        bx = x1r - ax * x1

        ay = (y2r - y1r) / (y2p[1] - y1p[1])
        by = y1r - ay * y1p[1]

        return ax * px + bx, ay * py + by


    # ---------------------------------------------------------
    # AUTODETECCIÓN

    def run_autodetect(self):
        if self.curve_color is None:
            print("Selecciona un color primero.")
            return

        base_thr = 0.08
        adapt = np.mean(self.curve_color) * 0.5
        thr = base_thr + adapt + 0.10

        diff = np.linalg.norm(self.img[:, :, :3] - self.curve_color, axis=2)
        mask = diff < thr

        ys, xs = np.where(mask)

        self.points_auto = []
        self.points_all = []
        self.internal_mode = None

        self.ax.clear()
        self.ax.imshow(self.img)

        for px, py in zip(xs, ys):
            real = self.pixel_to_real(px, py)
            if real:
                art = self.ax.plot(px, py, "o", markersize=5, color="yellow")[0]
                self.points_auto.append(real)
                self.points_all.append((px, py, real, art))

        self.fig.canvas.draw_idle()
        print("Auto detectó:", len(self.points_auto))


    # ---------------------------------------------------------
    # INTERPOLACIÓN

    def interpolate_curve(self, step=1.0):
        if not self.points_all:
            print("No hay puntos.")
            return []

        data = sorted([p for _,_,p,_ in self.points_all], key=lambda v: v[0])
        xs = np.array([p[0] for p in data])
        ys = np.array([p[1] for p in data])

        x_min, x_max = xs.min(), xs.max()
        x_new = np.arange(x_min, x_max, step)
        y_new = np.interp(x_new, xs, ys)

        return list(zip(x_new, y_new))


    # ---------------------------------------------------------
    # REINICIAR

    def reset_all(self):
        self.points_auto = []
        self.points_manual = []
        self.points_all = []
        self.curve_color = None
        self.internal_mode = None

        self.rect_start = None
        self.rect_patch = None

        self.axis_click_stage = 0
        self.calib = {"x1":None, "x2":None, "y1":None, "y2":None}

        # limpiar entradas
        self.x1_real.delete(0, tk.END)
        self.x2_real.delete(0, tk.END)
        self.y1_real.delete(0, tk.END)
        self.y2_real.delete(0, tk.END)

        self.x1_real.insert(0, "0")
        self.x2_real.insert(0, "1")
        self.y1_real.insert(0, "0")
        self.y2_real.insert(0, "1")

        # reinicio gráfico
        if hasattr(self, "img"):
            self.ax.clear()
            self.ax.imshow(self.img)
            self.fig.canvas.draw_idle()

        self.label_min.config(text="---")
        self.label_max.config(text="---")


    # ---------------------------------------------------------
    # EXPORTAR CSV

    def export_csv(self):
        data = self.interpolate_curve(step=1.0)

        #Sin interpolación
        #data = [p for _,_,p,_ in self.points_all]

        if not data:
            print("No hay puntos exportables.")
            return

        # Convertir lista → DataFrame
        df = pd.DataFrame(data, columns=["x", "y"])
        
        #Ordenar
        #df.sort_values(by="x").reset_index(drop=True)

        # Guardar CSV en formato inglés (solo permite .csv)
        path = filedialog.asksaveasfilename(
            title="Guardar archivo CSV",
            filetypes=[("CSV files", "*.csv")],
            defaultextension=".csv"
        )
        if not path:
            return

        # UTF-8 con BOM para compatibilidad con Excel
        df.to_csv(path, sep=",", index=False, encoding="utf-8-sig")

        # Calcular min y max
        pmin = df.loc[df["y"].idxmin()]
        pmax = df.loc[df["y"].idxmax()]

        self.label_min.config(text=f"{pmin['x']:.4f}, {pmin['y']:.4f}")
        self.label_max.config(text=f"{pmax['x']:.4f}, {pmax['y']:.4f}")

        print("Exportado correctamente:", path)

# ============================================================
# EJECUCIÓN
# ============================================================

root = tk.Tk()
DigitizerApp(root)
root.mainloop()
