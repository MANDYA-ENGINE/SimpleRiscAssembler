import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from final import SimpleRiscAssembler
import os
import sys
import tempfile
from threading import Thread

class SimpleRiscAssemblerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("S-RISC-21 Assembler")
        
        # Create toolbar
        toolbar = ttk.Frame(root)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # Toolbar buttons
        ttk.Button(toolbar, text="New", command=self.new_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Open", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="▶ Assemble", command=self.assemble_code).pack(side=tk.LEFT, padx=2)
        
        # Main content area with Panedwindow
        main_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        # Left panel - Assembly Code
        left_frame = ttk.Frame(main_pane)
        ttk.Label(left_frame, text="Assembly Code").pack(anchor=tk.W)
        self.editor = scrolledtext.ScrolledText(left_frame, wrap=tk.NONE, font=("Consolas", 11))
        self.editor.pack(fill=tk.BOTH, expand=True)
        main_pane.add(left_frame, weight=1)
        
        # Right panel - Output
        right_frame = ttk.Frame(main_pane)
        ttk.Label(right_frame, text="Assembled Output").pack(anchor=tk.W)
        
        # Notebook for different output views
        output_notebook = ttk.Notebook(right_frame)
        output_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Binary output tab
        self.binary_output = scrolledtext.ScrolledText(output_notebook, wrap=tk.NONE, 
                                                     font=("Consolas", 11), bg='black', fg='yellow')
        output_notebook.add(self.binary_output, text="Binary")
        
        # Hexadecimal output tab
        self.hex_output = scrolledtext.ScrolledText(output_notebook, wrap=tk.NONE, 
                                                  font=("Consolas", 11), bg='black', fg='yellow')
        output_notebook.add(self.hex_output, text="Hexadecimal")
        
        # Messages tab
        self.messages = scrolledtext.ScrolledText(output_notebook, wrap=tk.WORD, 
                                                font=("Consolas", 11), bg='black', fg='white')
        output_notebook.add(self.messages, text="Messages")
        
        main_pane.add(right_frame, weight=1)
        
        # Initialize assembler
        self.assembler = None
        self.input_file = None
        self.is_assembling = False
        self.temp_dir = tempfile.gettempdir()
    
    def new_file(self):
        if self.is_assembling:
            return
        if messagebox.askyesno("New File", "Create new file? Unsaved changes will be lost."):
            self.editor.delete("1.0", tk.END)
            self.input_file = None
    
    def open_file(self):
        if self.is_assembling:
            return
        file_path = filedialog.askopenfilename(
            title="Open Assembly File",
            filetypes=[("Assembly Files", "*.s *.asm"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.editor.delete("1.0", tk.END)
                    self.editor.insert("1.0", file.read())
                self.input_file = file_path
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {str(e)}")
    
    def save_file(self):
        if self.is_assembling:
            return
        if not self.input_file:
            file_path = filedialog.asksaveasfilename(
                title="Save Assembly File",
                defaultextension=".s",
                filetypes=[("Assembly Files", "*.s *.asm"), ("All Files", "*.*")]
            )
            if not file_path:
                return False
            self.input_file = file_path
        
        try:
            with open(self.input_file, 'w') as file:
                file.write(self.editor.get("1.0", "end-1c"))
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {str(e)}")
            return False
    
    def assemble_code(self):
        if self.is_assembling:
            return
        
        self.is_assembling = True
        self.clear_output()
        
        # Get editor content
        content = self.editor.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showerror("Error", "No code to assemble!")
            self.is_assembling = False
            return
        
        # Create temp file
        fd, input_file = tempfile.mkstemp(suffix='.s', text=True)
        os.close(fd)
        output_file = os.path.join(self.temp_dir, "output.bin")
        
        try:
            with open(input_file, 'w') as f:
                f.write(content)
        except Exception as e:
            messagebox.showerror("Error", f"Could not create temporary file: {str(e)}")
            self.is_assembling = False
            if os.path.exists(input_file):
                try:
                    os.remove(input_file)
                except:
                    pass
            return
        
        # Start assembly in a separate thread
        def assemble_thread():
            try:
                # Redirect stdout to capture messages
                old_stdout = sys.stdout
                sys.stdout = MessageCapture(self.messages)
                
                # Create new assembler instance
                self.assembler = SimpleRiscAssembler()
                self.assembler.assemble(input_file, output_file)
                
                # Display the output
                if os.path.exists(output_file):
                    # Read and display binary
                    with open(output_file, 'rb') as f:
                        binary_data = f.read()
                        self.binary_output.delete("1.0", tk.END)
                        for i, word in enumerate(range(0, len(binary_data), 4)):
                            binary = int.from_bytes(binary_data[word:word+4], 'little')
                            addr = format(i * 4, '08x')
                            self.binary_output.insert(tk.END, f"{addr}: {format(binary, '032b')}\n")
                    
                    # Read and display hex
                    hex_file = output_file.replace('.bin', '.hex')
                    if os.path.exists(hex_file):
                        with open(hex_file, 'r') as f:
                            self.hex_output.delete("1.0", tk.END)
                            self.hex_output.insert("1.0", f.read())
            
            except Exception as e:
                self.messages.insert(tk.END, f"\nError during assembly: {str(e)}\n")
            
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                
                # Clean up temp files
                if os.path.exists(input_file):
                    try:
                        os.remove(input_file)
                    except:
                        pass
                
                # Re-enable assembly
                self.is_assembling = False
        
        Thread(target=assemble_thread).start()
    
    def clear_output(self):
        """Clear all output areas."""
        self.binary_output.delete("1.0", tk.END)
        self.hex_output.delete("1.0", tk.END)
        self.messages.delete("1.0", tk.END)
    
    def insert_sample_code(self):
        """Insert sample code into the editor."""
        sample_code = """mov r1, 31
mov r2, 29
div r3,r1,r2
sub r4,r3,50"""
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", sample_code)

class MessageCapture:
    """Capture stdout and redirect to message widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget
    
    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
    
    def flush(self):
        pass

def main():
    root = tk.Tk()
    app = SimpleRiscAssemblerGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

if __name__ == "__main__":
    main()