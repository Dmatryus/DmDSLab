"""
Минималистичный GUI для экспериментов.
Позволяет настроить и запустить Synthdator и Benchmark.
"""

import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path
from dataclasses import fields
from typing import Any

# Добавляем пути к модулям
sys.path.insert(0, str(Path(__file__).parent / "synthdator"))
sys.path.insert(0, str(Path(__file__).parent / "dataframe_bench"))


class LogRedirector:
    """Перенаправляет stdout/stderr в Text виджет."""

    def __init__(self, text_widget: scrolledtext.ScrolledText):
        self.text_widget = text_widget

    def write(self, message: str) -> None:
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self) -> None:
        pass


class ToolTip:
    """Простой tooltip для виджетов."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None) -> None:
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("TkDefaultFont", 9), padx=5, pady=3
        )
        label.pack()

    def _hide(self, event=None) -> None:
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class ConfigFrame(ttk.LabelFrame):
    """Фрейм для редактирования dataclass конфига."""

    def __init__(self, parent: tk.Widget, config_class: type, title: str):
        super().__init__(parent, text=title, padding=10)
        self.config_class = config_class
        self.widgets: dict[str, tk.Widget] = {}
        self.field_docs = self._parse_docstring()
        self._create_widgets()

    def _parse_docstring(self) -> dict[str, str]:
        """Извлекает описания полей из docstring класса."""
        docs = {}
        docstring = self.config_class.__doc__ or ""

        # Ищем секцию Attributes
        in_attrs = False
        current_field = None
        current_desc = []

        for line in docstring.split("\n"):
            stripped = line.strip()

            if stripped.lower().startswith("attributes:"):
                in_attrs = True
                continue

            if in_attrs:
                # Проверяем отступ - поля имеют 8 пробелов, продолжение 12+
                indent = len(line) - len(line.lstrip())

                # Новое поле - 8 пробелов отступа и содержит ":"
                if indent == 8 and ":" in stripped:
                    # Сохраняем предыдущее
                    if current_field:
                        docs[current_field] = " ".join(current_desc).strip()
                    # Парсим новое поле
                    parts = stripped.split(":", 1)
                    current_field = parts[0].strip()
                    current_desc = [parts[1].strip()] if len(parts) > 1 else []
                elif indent > 8 and current_field and stripped:
                    # Продолжение описания (больше 8 пробелов)
                    current_desc.append(stripped)
                elif not stripped:
                    # Пустая строка может быть между полями
                    pass
                elif indent < 8 and stripped:
                    # Меньше отступа - конец секции Attributes
                    if current_field:
                        docs[current_field] = " ".join(current_desc).strip()
                    break

        # Не забываем последнее поле
        if current_field:
            docs[current_field] = " ".join(current_desc).strip()

        return docs

    def _create_widgets(self) -> None:
        """Создаёт виджеты для каждого поля конфига."""
        row = 0
        for f in fields(self.config_class):
            # Пропускаем сложные типы
            if "list" in str(f.type).lower() and "str" not in str(f.type).lower():
                continue

            # Label с именем поля
            label = ttk.Label(self, text=f.name)
            label.grid(row=row, column=0, sticky="w", padx=5, pady=2)

            # Добавляем tooltip с описанием
            description = self.field_docs.get(f.name, "")
            if description:
                ToolTip(label, description)

            # Определяем тип виджета
            widget = self._create_field_widget(f)
            widget.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            self.widgets[f.name] = widget

            # Tooltip и для виджета
            if description:
                ToolTip(widget, description)

            row += 1

        self.columnconfigure(1, weight=1)

    def _get_default_value(self, f) -> Any:
        """Получает значение по умолчанию для поля dataclass."""
        from dataclasses import MISSING

        # Если есть обычное значение по умолчанию
        if f.default is not MISSING:
            return f.default

        # Если есть default_factory
        if f.default_factory is not MISSING:
            try:
                return f.default_factory()
            except Exception:
                return None

        return None

    def _create_field_widget(self, f) -> tk.Widget:
        """Создаёт виджет для поля в зависимости от типа."""
        type_str = str(f.type).lower()
        default = self._get_default_value(f)

        # Boolean
        if "bool" in type_str:
            var = tk.BooleanVar(value=bool(default) if default is not None else False)
            widget = ttk.Checkbutton(self, variable=var)
            widget.var = var
            return widget

        # Path с кнопкой выбора
        if "path" in type_str:
            frame = ttk.Frame(self)
            var = tk.StringVar(value=str(default) if default else "")
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(side="left", fill="x", expand=True)

            def browse(v=var):
                path = filedialog.askdirectory()
                if path:
                    v.set(path)

            btn = ttk.Button(frame, text="...", width=3, command=browse)
            btn.pack(side="right", padx=(5, 0))
            frame.var = var
            return frame

        # Числа
        if "int" in type_str:
            var = tk.StringVar(value=str(default) if default is not None else "")
            widget = ttk.Entry(self, textvariable=var, width=15)
            widget.var = var
            return widget

        if "float" in type_str:
            var = tk.StringVar(value=str(default) if default is not None else "")
            widget = ttk.Entry(self, textvariable=var, width=15)
            widget.var = var
            return widget

        # Literal (выпадающий список)
        if "literal" in type_str:
            # Извлекаем варианты из Literal
            import re
            matches = re.findall(r"'([^']+)'", type_str)
            if matches:
                var = tk.StringVar(value=default if default else matches[0])
                widget = ttk.Combobox(self, textvariable=var, values=matches, state="readonly")
                widget.var = var
                return widget

        # Списки строк - показываем через запятую
        if "list" in type_str and "str" in type_str:
            list_val = default if isinstance(default, list) else []
            var = tk.StringVar(value=", ".join(list_val) if list_val else "")
            widget = ttk.Entry(self, textvariable=var)
            widget.var = var
            return widget

        # Строки и остальное
        default_val = ""
        if default is not None:
            try:
                default_val = str(default)
            except Exception:
                pass
        var = tk.StringVar(value=default_val)
        widget = ttk.Entry(self, textvariable=var)
        widget.var = var
        return widget

    def get_config(self) -> Any:
        """Собирает значения из виджетов и создаёт конфиг."""
        kwargs = {}
        for f in fields(self.config_class):
            if f.name not in self.widgets:
                continue

            widget = self.widgets[f.name]
            var = widget.var
            value = var.get()

            # Конвертируем типы
            type_str = str(f.type).lower()

            if value == "" or value == "None":
                kwargs[f.name] = None
                continue

            try:
                if "bool" in type_str:
                    kwargs[f.name] = bool(value)
                elif "int" in type_str and "none" not in type_str:
                    kwargs[f.name] = int(value)
                elif "int" in type_str:
                    kwargs[f.name] = int(value) if value else None
                elif "float" in type_str:
                    kwargs[f.name] = float(value)
                elif "path" in type_str:
                    kwargs[f.name] = Path(value) if value else None
                elif "list" in type_str and "str" in type_str:
                    # Парсим список строк через запятую
                    items = [s.strip() for s in value.split(",") if s.strip()]
                    kwargs[f.name] = items if items else None
                else:
                    kwargs[f.name] = value
            except ValueError:
                kwargs[f.name] = value

        return self.config_class(**kwargs)


class SynthdatorTab(ttk.Frame):
    """Вкладка для генерации данных."""

    def __init__(self, parent: tk.Widget, log_callback):
        super().__init__(parent, padding=10)
        self.log_callback = log_callback
        self._running = False
        self._create_widgets()

    def _create_widgets(self) -> None:
        # Импортируем конфиг
        try:
            from generator import GeneratorConfig
            self.config_frame = ConfigFrame(self, GeneratorConfig, "Generator Config")
            self.config_frame.pack(fill="both", expand=True)
        except ImportError as e:
            ttk.Label(self, text=f"Cannot import GeneratorConfig: {e}").pack()
            return

        # Кнопка запуска
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=10)

        self.run_btn = ttk.Button(btn_frame, text="Generate", command=self._run)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

    def _run(self) -> None:
        if self._running:
            return

        self._running = True
        self.run_btn.configure(state="disabled")
        self.progress.start()

        def task():
            try:
                from generator import GeneratorConfig, PipelineFactory

                config = self.config_frame.get_config()
                self.log_callback(f"Starting generation with config:\n{config}\n")

                factory = PipelineFactory()
                pipeline = factory.create(config)
                meta = pipeline.run()

                self.log_callback(f"\nGeneration complete!")
                self.log_callback(f"File: {meta.file_path}")
                self.log_callback(f"Rows: {meta.row_count}")
                self.log_callback(f"Columns: {list(meta.columns.keys())}\n")

            except Exception as e:
                self.log_callback(f"\nError: {e}\n")
            finally:
                self._running = False
                self.after(0, lambda: self.run_btn.configure(state="normal"))
                self.after(0, self.progress.stop)

        threading.Thread(target=task, daemon=True).start()


class BenchmarkTab(ttk.Frame):
    """Вкладка для запуска бенчмарков."""

    def __init__(self, parent: tk.Widget, log_callback):
        super().__init__(parent, padding=10)
        self.log_callback = log_callback
        self._running = False
        self._create_widgets()

    def _create_widgets(self) -> None:
        # Импортируем конфиг
        try:
            from polars_duckdb_bench import BenchmarkConfig
            self.config_frame = ConfigFrame(self, BenchmarkConfig, "Benchmark Config")
            self.config_frame.pack(fill="both", expand=True)
        except ImportError as e:
            ttk.Label(self, text=f"Cannot import BenchmarkConfig: {e}").pack()
            return

        # Кнопка запуска
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=10)

        self.run_btn = ttk.Button(btn_frame, text="Run Benchmarks", command=self._run)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

    def _run(self) -> None:
        if self._running:
            return

        self._running = True
        self.run_btn.configure(state="disabled")
        self.progress.start()

        def task():
            try:
                from polars_duckdb_bench import BenchmarkConfig, BenchmarkFactory

                config = self.config_frame.get_config()
                self.log_callback(f"Starting benchmarks with config:\n{config}\n")

                if config.data_path is None:
                    self.log_callback("Error: data_path is required!\n")
                    return

                factory = BenchmarkFactory()
                suite = factory.create(config)
                results = suite.run()

                self.log_callback(f"\n{results.summary()}\n")

            except Exception as e:
                self.log_callback(f"\nError: {e}\n")
                import traceback
                self.log_callback(traceback.format_exc())
            finally:
                self._running = False
                self.after(0, lambda: self.run_btn.configure(state="normal"))
                self.after(0, self.progress.stop)

        threading.Thread(target=task, daemon=True).start()


class ExperimentsGUI(tk.Tk):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        self.title("DmDSLab Experiments")
        self.geometry("800x600")
        self._create_widgets()

    def _create_widgets(self) -> None:
        # Основной контейнер с вкладками
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Лог внизу
        log_frame = ttk.LabelFrame(self, text="Output", padding=5)
        log_frame.pack(fill="x", padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # Вкладки
        self.synthdator_tab = SynthdatorTab(self.notebook, self._log)
        self.notebook.add(self.synthdator_tab, text="Synthdator")

        self.benchmark_tab = BenchmarkTab(self.notebook, self._log)
        self.notebook.add(self.benchmark_tab, text="Benchmark")

        # Кнопка очистки лога
        clear_btn = ttk.Button(log_frame, text="Clear", command=self._clear_log)
        clear_btn.pack(anchor="e", pady=5)

    def _log(self, message: str) -> None:
        """Добавляет сообщение в лог."""
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        """Очищает лог."""
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")


def main():
    """Запуск GUI."""
    app = ExperimentsGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
