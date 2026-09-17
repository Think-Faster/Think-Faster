import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Загрузка данных
# =========================

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dispatcher_data.xlsx"

df = pd.read_excel(file_path)

df.columns = df.columns.astype(str).str.strip()


# =========================
# Граф
# =========================

G = nx.DiGraph()

# Сначала добавляем ВСЕ объекты
for _, row in df.iterrows():

    object_id = row["ид_объект"]

    G.add_node(
        object_id,
        name=str(row["диспетчерское_название_объекта"]),
        type=str(row["вид_объекта"]),
        level=row["иерархия_уровень"]
    )


# Потом добавляем связи
for _, row in df.iterrows():

    parent = row["родитель"]
    object_id = row["ид_объект"]

    if pd.notna(parent):

        # Добавляем связь только если родитель существует
        if parent in G:
            G.add_edge(parent, object_id)


# =========================
# Иерархическое расположение
# =========================

def hierarchy_pos(G):

    pos = {}

    # Корневые элементы
    roots = [
        node
        for node in G.nodes
        if G.in_degree(node) == 0
    ]

    def calculate_width(node):

        children = list(G.successors(node))

        if not children:
            return 1

        return sum(
            calculate_width(child)
            for child in children
        )

    def place(node, x, y):

        children = list(G.successors(node))

        # Лист
        if not children:
            pos[node] = (x, y)
            return x + 1

        # Общая ширина дерева
        child_widths = [
            calculate_width(child)
            for child in children
        ]

        total_width = sum(child_widths)

        current_x = x

        child_positions = []

        for child, width in zip(children, child_widths):

            child_center = current_x + width / 2

            place(
                child,
                child_center,
                y - 1
            )

            child_positions.append(child_center)

            current_x += width

        # Родитель по центру детей
        pos[node] = (
            sum(child_positions) / len(child_positions),
            y
        )

        return x + total_width

    current_x = 0

    for root in roots:

        width = calculate_width(root)

        place(
            root,
            current_x + width / 2,
            0
        )

        current_x += width + 2

    return pos


pos = hierarchy_pos(G)


# =========================
# Подписи
# =========================

labels = {}

for node in G.nodes:

    name = G.nodes[node].get("name")

    if name:
        labels[node] = name
    else:
        # Если узел почему-то без названия
        labels[node] = str(node)


# =========================
# Размер картинки
# =========================

width = max(20, len(G.nodes) * 0.5)

plt.figure(
    figsize=(width, 16)
)


# =========================
# Рисуем граф
# =========================

nx.draw(
    G,
    pos,

    labels=labels,

    node_size=3000,

    arrows=True,
    arrowsize=20,

    width=1.5,

    font_size=8,

    node_shape="o"
)


plt.title(
    "Иерархия диспетчерских объектов",
    fontsize=18
)

plt.axis("off")

plt.tight_layout()

plt.show()