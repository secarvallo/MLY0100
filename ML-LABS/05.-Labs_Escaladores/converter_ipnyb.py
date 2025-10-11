import json
from pathlib import Path

def ipynb_to_py(ipynb_file, py_file):
    """
    Convierte un archivo .ipynb a .py
    """
    base_dir = Path(__file__).resolve().parent

    ipynb_path = Path(ipynb_file)
    if not ipynb_path.is_absolute():
        ipynb_path = base_dir / ipynb_path

    if not ipynb_path.exists():
        raise FileNotFoundError(f"No se encontró el notebook en {ipynb_path}")

    py_path = Path(py_file)
    if not py_path.is_absolute():
        py_path = base_dir / py_path

    py_path.parent.mkdir(parents=True, exist_ok=True)

    with ipynb_path.open('r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    with py_path.open('w', encoding='utf-8') as f:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                # Escribir código
                for line in cell['source']:
                    f.write(line)
                f.write('\n\n')
            elif cell['cell_type'] == 'markdown':
                # Convertir markdown a comentarios
                f.write('\n# ' + '='*50 + '\n')
                for line in cell['source']:
                    f.write('# ' + line)
                f.write('\n# ' + '='*50 + '\n\n')

# Uso
ipynb_to_py('PARDO_DIAZ_MEJIAS_CHANILLAO_CARVALLO_LABSN05.ipynb', 'mi_script.py')