import customtkinter as ctk

# Set global appearance
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class ScenarioConfigDialog(ctk.CTkFrame):
    def __init__(self, parent, on_run=None, on_cancel=None):
        super().__init__(
            parent,
            corner_radius=20,
            fg_color="#FFFFFF",
            border_color="#E5E7EB",
            border_width=1,
        )
        self.on_run = on_run
        self.on_cancel = on_cancel

        # Data options
        self.planner_algos = ["CBS", "M*", "Voronoi"]
        self.marl_algos = ["QTRAN", "QPLEX", "QMIX", "VDN"]
        self.game_modes = ["Maze", "Reach and Avoid", "Pure Pursuit"]

        # Tracked Variables
        self.walls_density_val = 0.5
        self.barrels_density_val = 0.0

        self._build_ui()

    def _build_ui(self):
        # Container padding
        self.grid_columnconfigure(0, weight=1)
        
        # 1. Title & Subtitle
        title_label = ctk.CTkLabel(
            self,
            text="Scenario Configuration",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#1E1E2F",
            anchor="w",
        )
        title_label.pack(fill="x", padx=30, pady=(25, 2))

        subtitle_label = ctk.CTkLabel(
            self,
            text="Set up your environment and algorithm parameters.",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#6B7280",
            anchor="w",
        )
        subtitle_label.pack(fill="x", padx=30, pady=(0, 20))

        # 2. Game Mode Dropdown
        self._section_label("GAME MODE")
        self.mode_dropdown = ctk.CTkOptionMenu(
            self,
            values=self.game_modes,
            command=self._on_mode_change,
            fg_color="#FFFFFF",
            text_color="#111827",
            button_color="#FFFFFF",
            button_hover_color="#F3F4F6",
            dropdown_fg_color="#FFFFFF",
            dropdown_text_color="#111827",
            corner_radius=10,
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.mode_dropdown.pack(fill="x", padx=30, pady=(0, 4))

        mode_help = ctk.CTkLabel(
            self,
            text="Maze · Reach and Avoid · Pure Pursuit",
            font=ctk.CTkFont(size=11),
            text_color="#9CA3AF",
            anchor="w",
        )
        mode_help.pack(fill="x", padx=30, pady=(0, 16))

        # 3. Algorithm Family Toggle
        self._section_label("ALGORITHM")
        self.family_toggle = ctk.CTkSegmentedButton(
            self,
            values=["Planner", "MARL"],
            command=self._on_family_change,
            fg_color="#F3F4F6",
            selected_color="#1C0056",
            selected_hover_color="#2D0080",
            unselected_color="#FFFFFF",
            unselected_hover_color="#F9FAFB",
            text_color="#111827",
            corner_radius=10,
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.family_toggle.set("Planner")
        self.family_toggle.pack(fill="x", padx=30, pady=(0, 16))

        # 4. Specific Algorithm Selection
        self.algo_section_label = ctk.CTkLabel(
            self,
            text="PLANNER ALGORITHM",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#6B7280",
            anchor="w",
        )
        self.algo_section_label.pack(fill="x", padx=30, pady=(0, 6))

        self.algo_dropdown = ctk.CTkOptionMenu(
            self,
            values=self.planner_algos,
            fg_color="#FFFFFF",
            text_color="#111827",
            button_color="#FFFFFF",
            button_hover_color="#F3F4F6",
            dropdown_fg_color="#FFFFFF",
            dropdown_text_color="#111827",
            corner_radius=10,
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.algo_dropdown.pack(fill="x", padx=30, pady=(0, 4))

        self.algo_help_label = ctk.CTkLabel(
            self,
            text="Planner: CBS, Voronoi, M*\nMARL: QTRAN, QPLEX, QMIX, VDN\nOnly the selected family's algorithms are shown above.",
            font=ctk.CTkFont(size=11),
            text_color="#9CA3AF",
            anchor="w",
            justify="left",
        )
        self.algo_help_label.pack(fill="x", padx=30, pady=(0, 18))

        # 5. Scenario Size Input & Resulting Card
        self._section_label("SCENARIO SIZE (X = Y)")
        size_frame = ctk.CTkFrame(self, fg_color="transparent")
        size_frame.pack(fill="x", padx=30, pady=(0, 18))

        self.size_entry = ctk.CTkEntry(
            size_frame,
            placeholder_text="20",
            fg_color="#FFFFFF",
            text_color="#111827",
            border_color="#D1D5DB",
            border_width=1,
            corner_radius=10,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.size_entry.insert(0, "20")
        self.size_entry.pack(side="left", expand=True, fill="x", padx=(0, 12))
        self.size_entry.bind("<KeyRelease>", self._update_size_badge)

        units_tag = ctk.CTkLabel(
            self.size_entry,
            text="units",
            text_color="#9CA3AF",
            font=ctk.CTkFont(size=12),
        )
        units_tag.place(relx=0.92, rely=0.5, anchor="e")

        self.badge_card = ctk.CTkFrame(
            size_frame,
            fg_color="#FFFFFF",
            border_color="#E5E7EB",
            border_width=1,
            corner_radius=10,
            width=130,
            height=45,
        )
        self.badge_card.pack(side="right")
        self.badge_card.pack_propagate(False)

        badge_sub = ctk.CTkLabel(
            self.badge_card,
            text="Resulting size",
            font=ctk.CTkFont(size=10),
            text_color="#6B7280",
        )
        badge_sub.pack(anchor="w", padx=10, pady=(4, 0))

        self.badge_val = ctk.CTkLabel(
            self.badge_card,
            text="20×20",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#111827",
        )
        self.badge_val.pack(anchor="w", padx=10, pady=(0, 4))

        # 6. Object Density
        self._section_label("OBJECT DENSITY")

        # Walls Row
        self.walls_row = self._create_stepper_row(
            "Walls Density", self.walls_density_val, self._on_walls_change
        )
        self.walls_row.pack(fill="x", padx=30, pady=(0, 8))

        # Barrels Row (Disabled in Maze)
        self.barrels_row, self.barrels_val_label, self.barrel_btns = (
            self._create_stepper_row_control(
                "Barrels Density", self.barrels_density_val, self._on_barrels_change
            )
        )
        self.barrels_row.pack(fill="x", padx=30, pady=(0, 8))

        # Info Banner
        self.info_banner = ctk.CTkLabel(
            self,
            text="ⓘ  Barrels are unavailable in Maze mode, so density is fixed at 0.",
            font=ctk.CTkFont(size=11),
            text_color="#9CA3AF",
            anchor="w",
        )
        self.info_banner.pack(fill="x", padx=30, pady=(0, 25))

        # Initial Mode Check
        self._on_mode_change("Maze")

        # 7. Action Footer Buttons
        footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        footer_frame.pack(fill="x", padx=30, pady=(0, 25))

        run_btn = ctk.CTkButton(
            footer_frame,
            text="Run Scenario",
            command=self._handle_run,
            fg_color="#1C0056",
            hover_color="#2D0080",
            text_color="#FFFFFF",
            corner_radius=10,
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            width=120,
        )
        run_btn.pack(side="right", padx=(8, 0))

        cancel_btn = ctk.CTkButton(
            footer_frame,
            text="Cancel",
            command=self.on_cancel,
            fg_color="#FFFFFF",
            hover_color="#F3F4F6",
            text_color="#111827",
            border_color="#D1D5DB",
            border_width=1,
            corner_radius=10,
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            width=90,
        )
        cancel_btn.pack(side="right")

    def _section_label(self, text):
        lbl = ctk.CTkLabel(
            self,
            text=text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#6B7280",
            anchor="w",
        )
        lbl.pack(fill="x", padx=30, pady=(0, 6))

    def _on_family_change(self, choice):
        if choice == "Planner":
            self.algo_section_label.configure(text="PLANNER ALGORITHM")
            self.algo_dropdown.configure(values=self.planner_algos)
            self.algo_dropdown.set(self.planner_algos[0])
        else:
            self.algo_section_label.configure(text="MARL ALGORITHM")
            self.algo_dropdown.configure(values=self.marl_algos)
            self.algo_dropdown.set(self.marl_algos[0])

    def _on_mode_change(self, choice):
        if choice == "Maze":
            # Barrels locked at 0 in Maze
            self.barrels_val_label.configure(text="0.0", text_color="#D1D5DB")
            for btn in self.barrel_btns:
                btn.configure(
                    state="disabled", fg_color="#F9FAFB", text_color="#D1D5DB"
                )
            self.info_banner.pack(fill="x", padx=30, pady=(0, 25))
        else:
            self.barrels_val_label.configure(
                text=f"{self.barrels_density_val:.1f}", text_color="#111827"
            )
            for btn in self.barrel_btns:
                btn.configure(
                    state="normal", fg_color="#FFFFFF", text_color="#111827"
                )
            self.info_banner.pack_forget()

    def _update_size_badge(self, event=None):
        val = self.size_entry.get().strip()
        if val.isdigit():
            self.badge_val.configure(text=f"{val}×{val}")

    def _create_stepper_row(self, title, initial_val, callback):
        row, _, _ = self._create_stepper_row_control(title, initial_val, callback)
        return row

    def _create_stepper_row_control(self, title, initial_val, callback):
        row = ctk.CTkFrame(self, fg_color="transparent")

        lbl = ctk.CTkLabel(
            row,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#111827",
            anchor="w",
        )
        lbl.pack(side="left", expand=True, fill="x")

        stepper = ctk.CTkFrame(row, fg_color="transparent")
        stepper.pack(side="right")

        minus_btn = ctk.CTkButton(
            stepper,
            text="-",
            width=32,
            height=32,
            fg_color="#FFFFFF",
            hover_color="#F3F4F6",
            text_color="#111827",
            border_color="#E5E7EB",
            border_width=1,
            corner_radius=6,
            command=lambda: callback(-0.1),
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        minus_btn.pack(side="left", padx=2)

        val_box = ctk.CTkFrame(
            stepper,
            fg_color="#FFFFFF",
            border_color="#E5E7EB",
            border_width=1,
            corner_radius=6,
            width=50,
            height=32,
        )
        val_box.pack(side="left", padx=2)
        val_box.pack_propagate(False)

        val_lbl = ctk.CTkLabel(
            val_box,
            text=f"{initial_val:.1f}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#111827",
        )
        val_lbl.place(relx=0.5, rely=0.5, anchor="center")

        plus_btn = ctk.CTkButton(
            stepper,
            text="+",
            width=32,
            height=32,
            fg_color="#FFFFFF",
            hover_color="#F3F4F6",
            text_color="#111827",
            border_color="#E5E7EB",
            border_width=1,
            corner_radius=6,
            command=lambda: callback(0.1),
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        plus_btn.pack(side="left", padx=2)

        return row, val_lbl, [minus_btn, plus_btn]

    def _on_walls_change(self, delta):
        self.walls_density_val = round(
            max(0.0, min(1.0, self.walls_density_val + delta)), 1
        )
        # Update UI label ...

    def _on_barrels_change(self, delta):
        if self.mode_dropdown.get() != "Maze":
            self.barrels_density_val = round(
                max(0.0, min(1.0, self.barrels_density_val + delta)), 1
            )
            self.barrels_val_label.configure(
                text=f"{self.barrels_density_val:.1f}"
            )

    def _handle_run(self):
        config = {
            "mode": self.mode_dropdown.get(),
            "family": self.family_toggle.get(),
            "algorithm": self.algo_dropdown.get(),
            "size": self.size_entry.get(),
            "walls_density": self.walls_density_val,
            "barrels_density": (
                0.0
                if self.mode_dropdown.get() == "Maze"
                else self.barrels_density_val
            ),
        }
        if self.on_run:
            self.on_run(config)


# App Launcher Demo
if __name__ == "__main__":
    app = ctk.CTk()
    app.title("Thesis Pursuit-Evasion Simulator")
    app.geometry("600, 750")
    app.configure(fg_color="#F3F4F6")

    def print_config(config):
        print("Starting Simulation with Config:", config)

    modal = ScenarioConfigDialog(
        app, on_run=print_config, on_cancel=app.destroy
    )
    modal.pack(padx=20, pady=20, fill="both", expand=True)

    app.mainloop()