"""Build the thin Colab launcher (training stays in ru_kws)."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "id": f"cell-{len(cells):02d}",
            "source": dedent(source).strip() + "\n"}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add("markdown", '''
# ru-kws: обучение BC-ResNet в Colab

Блокнот запускает модули проекта; обучение не дублируется в ячейках.

1. Включите GPU в настройках среды Colab.
2. Проверьте пути в настройках ниже.
3. Выполните ячейки до validation-отчёта.

Код загружается из [mole88/ru-kws](https://github.com/mole88/ru-kws), ветка `master`.
Загрузите датасет `russian_commands_v001_clean_manual.zip` в корень «Мой диск».
Для повторения эксперимента можно указать commit SHA в `REPO_REF`.

WAV распаковываются на локальный диск Colab; checkpoints и отчёты пишутся на Drive.
Baseline: 3 секунды, BC-ResNet-8, случайное размещение аудио, без noise mixing и
SpecAugment. При отключении Colab завершённые эпохи сохранятся, но **resume пока нет**.
Не используйте новый запуск обучения как продолжение старого.
''')
add("code", '''
from pathlib import Path
import sys, os, json, subprocess, hashlib, zipfile, tempfile, stat, csv

REPO_URL = "https://github.com/mole88/ru-kws.git"
REPO_REF = "master"       # Ветка, tag или commit SHA; SHA удобнее для повторяемости
DATASET_ZIP = Path("/content/drive/MyDrive/russian_commands_v001_clean_manual.zip")

RUN_NAME = "bcresnet_run_002"  # Новое имя для каждого обучения
RUNS_ROOT = Path("/content/drive/MyDrive/russian_commands/runs")
WINDOW_SECONDS = 3.0
BASE_C = 64                   # 8/12/16/24/48/64; 64 соответствует BC-ResNet-8
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 3e-4
REQUIRE_GPU = True            # False только для намеренного CPU запуска

# Финальный test выключен, чтобы не использовать его для подбора параметров.
RUN_FINAL_TEST = False

if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
    raise ValueError("RUN_NAME должен быть именем папки без пути")
RUN_DIR = RUNS_ROOT / RUN_NAME
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["MPLBACKEND"] = "Agg"
print("Результаты:", RUN_DIR)
''')
add("markdown", "## 1. Подключение Google Drive")
add("code", '''
from google.colab import drive
drive.mount("/content/drive")
''')
add("markdown", "## 2. Загрузка кода проекта\nКлонируем GitHub-репозиторий и фиксируем SHA загруженной версии кода.")
add("code", '''
def run(command, cwd=None):
    print("Запуск:", " ".join(map(str, command)), flush=True)
    with subprocess.Popen(list(map(str, command)), cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Команда завершилась с кодом {process.returncode}; ошибка выше")

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def extract_cached(archive, parent):
    if not archive.is_file():
        raise FileNotFoundError(f"Загрузите архив на Drive или исправьте путь: {archive}")
    digest = file_sha256(archive)
    target = parent / digest
    marker = target / ".extraction_complete"
    if marker.exists():
        print("Используем уже распакованный архив:", target)
        return target, digest
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in members:
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Недопустимый путь в ZIP: {member.filename}")
        print(f"Распаковка {len(members)} файлов...", flush=True)
        zf.extractall(target)
    marker.write_text(digest)
    return target, digest

def find_root(parent, marker, required):
    candidates = [p.parent for p in parent.rglob(marker)
                  if "__MACOSX" not in p.parts and all((p.parent / item).exists() for item in required)]
    if len(candidates) != 1:
        raise ValueError(f"Ожидалась одна папка с {marker}, найдены: {candidates}")
    return candidates[0]

if not REPO_URL.startswith("https://github.com/"):
    raise ValueError("Укажите HTTPS URL GitHub-репозитория в REPO_URL")
REPO = Path(tempfile.mkdtemp(prefix="ru-kws-", dir="/content"))
run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, REPO])
run(["git", "fetch", "--depth", "1", "origin", REPO_REF], cwd=REPO)
run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO)
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
source_info = {"mode": "github", "url": REPO_URL, "requested_ref": REPO_REF, "revision": revision}
print("Проект:", REPO)
print(json.dumps(source_info, ensure_ascii=False, indent=2))
''')
add("markdown", '''
## 3. Установка и проверка среды

Используем PyTorch/torchaudio среды Colab; эта ячейка не переустанавливает их.
Чтение WAV реализовано через SoundFile и не требует `torchcodec`.
Если проверка GPU не проходит, включите GPU и перезапустите среду.
''')
add("code", '''
run([sys.executable, "-m", "pip", "install", "numpy>=1.24", "soundfile>=0.12", "PyYAML>=6", "matplotlib>=3.6", "packaging"])
run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", REPO])
probe = """
import sys, json, torch, torchaudio
import ru_kws.data.validate, ru_kws.train, ru_kws.evaluate
from packaging.version import Version
from ru_kws.audio.frontend import LogMelFrontend
from ru_kws.models.bcresnet import BCResNets
assert Version(torch.__version__.split('+')[0]) >= Version('2.5')
assert Version(torchaudio.__version__.split('+')[0]) >= Version('2.5')
assert not REQUIRE_GPU or torch.cuda.is_available(), 'Включите GPU в настройках Colab'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with torch.inference_mode():
    x = LogMelFrontend().to(device)(torch.zeros(1, 48000, device=device))
    model = BCResNets(8, num_classes=8).to(device).eval()
    y = model(x)
assert x.shape == (1, 1, 40, 301) and y.shape == (1, 8)
print(json.dumps({'python': sys.version, 'torch': str(torch.__version__),
    'torchaudio': str(torchaudio.__version__), 'device': device,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))
"""
run([sys.executable, "-c", "REQUIRE_GPU = " + repr(REQUIRE_GPU) + "\\n" + probe])
''')
add("markdown", '''
## 4. Локальная копия датасета

Корень с `labels.json` находится автоматически, включая вложенную папку архива.
Повторная распаковка того же архива в одной среде пропускается.
''')
add("code", '''
unpacked, dataset_archive_hash = extract_cached(DATASET_ZIP, Path("/content/ru_kws_data"))
DATA_ROOT = find_root(unpacked, "labels.json", ["splits/train.jsonl", "splits/val.jsonl", "splits/test.jsonl"])
print("Датасет:", DATA_ROOT)
print("Метки:", json.loads((DATA_ROOT / "labels.json").read_text(encoding="utf-8")))
''')
add("markdown", '''
## 5. Конфигурация и проверка данных

Команды длиннее `WINDOW_SECONDS` автоматически исключаются из train/val/test.
Исходные WAV и manifests остаются без изменений. Число исключённых записей выводится
при проверке и загрузке данных; метрики считаются по оставшимся записям.
Длинные unknown/background сохраняются и обрезаются до окна при загрузке.
Валидатор всех splits проверяет формат и отсутствие пересечений, а не качество модели.
''')
add("code", '''
import yaml
config = yaml.safe_load((REPO / "configs/baseline.yaml").read_text(encoding="utf-8"))
config["audio"]["window_seconds"] = WINDOW_SECONDS
config["model"]["base_c"] = BASE_C
config["training"].update(batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                          learning_rate=LEARNING_RATE, num_workers=0)
CONFIG_PATH = Path("/content/ru_kws_colab_config.yaml")
CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(CONFIG_PATH.read_text())
run([sys.executable, "-m", "ru_kws.data.validate", "--data-root", DATA_ROOT,
     "--window-seconds", WINDOW_SECONDS], cwd=REPO)
''')
add("markdown", '''
## 6. Обучение

Начало эпохи может занимать время; loss/accuracy выводятся после каждой эпохи.
`best.pt` выбирается по validation loss, `last.pt` сохраняется после каждой эпохи.
Повторный запуск в непустую папку запрещён. Для нового эксперимента поменяйте `RUN_NAME`.
Чтобы только оценить существующий checkpoint, пропустите эту ячейку.
''')
add("code", '''
if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f"Папка уже содержит запуск: {RUN_DIR}. Выберите новый RUN_NAME или перейдите к оценке.")
run([sys.executable, "-m", "ru_kws.train", "--config", CONFIG_PATH,
     "--data-root", DATA_ROOT, "--run-dir", RUN_DIR, "--device", "auto"], cwd=REPO)
(RUN_DIR / "colab_source.json").write_text(json.dumps({
    "source": source_info, "dataset_archive": str(DATASET_ZIP),
    "dataset_archive_sha256": dataset_archive_hash,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Лучший checkpoint:", RUN_DIR / "best.pt")
''')
add("markdown", "## 7. Графики обучения\nСтроятся из сохранённого `history.csv`, без повторного обучения.")
add("code", '''
import matplotlib.pyplot as plt
with (RUN_DIR / "history.csv").open(encoding="utf-8") as stream:
    history = list(csv.DictReader(stream))
epochs = [int(row["epoch"]) for row in history]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for key in ("train_loss", "val_loss"):
    axes[0].plot(epochs, [float(row[key]) for row in history], label=key)
for key in ("train_accuracy", "val_accuracy"):
    axes[1].plot(epochs, [float(row[key]) for row in history], label=key)
for ax, title in zip(axes, ("Loss", "Accuracy")):
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
(RUN_DIR / "reports").mkdir(exist_ok=True)
fig.savefig(RUN_DIR / "reports" / "training.png", dpi=150)
plt.show()
''')
add("markdown", '''
## 8. Оценка лучшего checkpoint на validation

CLI сам загружает `best.pt`, его frontend config и label map.
Это classification-метрики клипов, не FP/hour и не задержка потокового распознавания.
''')
add("code", '''
def evaluate_split(split):
    checkpoint = RUN_DIR / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Не найден checkpoint: {checkpoint}")
    output_path = RUN_DIR / "reports" / f"{split}.json"
    run([sys.executable, "-m", "ru_kws.evaluate", "--checkpoint", checkpoint,
         "--data-root", DATA_ROOT, "--split", split, "--output", output_path], cwd=REPO)
    return json.loads(output_path.read_text(encoding="utf-8"))

def show_report(report):
    import numpy as np
    print(f"Epoch: {report['epoch']} | split: {report['split']} | N={report['count']}")
    print(f"Accuracy: {report['accuracy']:.4f} | macro F1: {report['macro_f1']:.4f}")
    print(f"{'Class':20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
    for name in report["class_order"]:
        row = report["per_class"][name]
        print(f"{name:20s} {row['precision']:10.4f} {row['recall']:10.4f} {row['f1']:10.4f} {row['support']:8d}")
    cm = np.array(report["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    names = report["class_order"]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black")
    ax.set_xlabel("Предсказанный класс"); ax.set_ylabel("Истинный класс")
    ax.set_title(f"{report['split']} — best checkpoint, epoch {report['epoch']}")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(RUN_DIR / "reports" / f"{report['split']}_confusion.png", dpi=150)
    plt.show()

val_report = evaluate_split("val")
show_report(val_report)
''')
add("markdown", '''
## 9. Финальный test — опционально

После фиксации модели и настроек поставьте `RUN_FINAL_TEST=True` в настройках
или непосредственно перед условием ниже. По умолчанию при «Выполнить все» test не запускается.
''')
add("code", '''
if RUN_FINAL_TEST:
    test_report = evaluate_split("test")
    show_report(test_report)
else:
    print("Test пропущен. Используйте validation для подбора настроек.")
print("Все результаты на Drive:", RUN_DIR)
''')

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "accelerator": "GPU",
    "colab": {"name": "ru_kws_training.ipynb", "provenance": []}},
    "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks" / "ru_kws_training.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
