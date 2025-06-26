import pandas as pd
import numpy as np
import re
import os
import json
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
from autoDataLakeL2 import fillDataLake
from sendEmailL import sendAlert

warnings.simplefilter("ignore")

# Константы
MEGA_NODE_PATH = os.path.join('/opt', 'otp', 'external_data', 'megaNode', 'megaNode.json')
MONTHES = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']

'''
class GraphProperty:

    def __init__(self, name, expression, value=None, status='pending'):
        self.name = name
        self.expression = expression
        self.value = value
        self.status = status
        
    def to_dict(self):
        return {
            'expression': self.expression,
            'value': self.value,
            'status': self.status
        }
'''
@dataclass
class GraphProperty:
    name: str
    expression: str
    value: any = None
    status: str = 'pending'

    def to_dict(self):
        return {
            'name': self.name,
            'expression': self.expression,
            'value': self.value,
            'status': self.status
        }       


@dataclass
class GraphNode:
    primitive_id: str
    primitive_name: str
    properties: dict = None

    def update_from_dict(self, node_data: dict):
        self.properties = node_data.get('properties', {})

    def add_property(self, name, expression, value=None, status='pending'):
        if self.properties is None:
            self.properties = {}
        self.properties[name] = GraphProperty(name, expression, value, status)

    def get_property(self, name):
        if self.properties is None:
            return None
        return self.properties.get(name)


'''       
class GraphNode:

    def __init__(self, primitive_id, primitive_name):
        self.primitive_id = primitive_id
        self.primitive_name = primitive_name
        self.properties = {}
        self.init_ports = []
        
    def add_property(self, name, expression, value=None, status='pending'):
        self.properties[name] = GraphProperty(name, expression, value, status)
        
    def get_property(self, name):
        return self.properties.get(name)
        
    def update_from_dict(self, node_data):
        for prop_name, prop_data in node_data['properties'].items():
            self.add_property(
                prop_name,
                prop_data['expression'],
                prop_data.get('value'),
                prop_data.get('status', 'pending')
            )
        self.init_ports = node_data.get('initPorts', [])
'''
@dataclass
class GraphConnection:
    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def to_dict(self):
        return {
            'source_node': self.source_node,
            'source_port': self.source_port,
            'target_node': self.target_node,
            'target_port': self.target_port
        }

'''        
class GraphConnection:

    def __init__(self, source_node, source_port, target_node, target_port):
        self.source_node = source_node
        self.source_port = source_port
        self.target_node = target_node
        self.target_port = target_port
        
    def to_dict(self):
        """Преобразует объект в словарь для сериализации в JSON"""
        return {
            'sourceNode': self.source_node,
            'targetNode': self.target_node,
            'sourcePort': self.source_port,
            'targetPort': self.target_port
        }
'''

        
class GraphStructure:

    def __init__(self, graph_json=None, node_class=GraphNode, connection_class=GraphConnection):
        self.nodes = {}
        self.connections = []
        self.node_class = node_class
        self.connection_class = connection_class
        
        if graph_json:
            self.load_from_json(graph_json)
    
    def load_from_json(self, graph_json):
        self._load_nodes(graph_json['graph']['nodes'])
        self._load_edges(graph_json['graph']['edges'])

    def _load_nodes(self, node_list):
        for node_data in node_list:
            node = node_class(
                node_data['primitiveID'], 
                node_data['primitiveName']
            )
            node.update_from_dict(node_data)
            self.nodes[node.primitive_id] = node

    def _load_edges(self, edge_list):
        for edge_data in edge_list:
            connection = connection_class(
                edge_data['sourceNode'],
                edge_data['sourcePort'],
                edge_data['targetNode'],
                edge_data['targetPort']
            )
            self.connections.append(connection)
    
    def to_json(self):
        nodes_json = []
        for node in self.nodes.values():
            node_data = {
                'primitiveID': node.primitive_id,
                'primitiveName': node.primitive_name,
                'properties': {name: prop.to_dict() for name, prop in node.properties.items()},
                'initPorts': node.init_ports
            }
            nodes_json.append(node_data)
        
        edges_json = [conn.to_dict() for conn in self.connections]
        
        return {'graph': {'nodes': nodes_json, 'edges': edges_json}}
    
    def get_node(self, node_id):
        return self.nodes.get(node_id)
    
    def add_node(self, node):
        self.nodes[node.primitive_id] = node
    
    def add_connection(self, connection):
        self.connections.append(connection)
        
#старый ExpressionCalculator###

# -----------------------------
# ExpressionContext: все зависимости для расчёта выражений
# -----------------------------
@dataclass
class ExpressionContext:
    swt: pd.DataFrame
    nodes: list
    node_numbers: Dict[str, int]
    dict_node_properties: Dict[str, str]

# -----------------------------
# ExpressionPreprocessor: нормализация выражения
# -----------------------------
class ExpressionPreprocessor:
    @staticmethod
    def preprocess(expression: str) -> str:
        if not isinstance(expression, str):
            return expression

        # Заменяем CalcX.value → SWT['X']
        expression = re.sub(r"Calc(.*?)\.value", r"SWT['\1']", expression)

        # Удаляем лишние пробелы и ifnull()
        if 'ifnull' in expression:
            expression = expression.replace('ifnull(', '').replace(' ', '')[:-1]

        # Замена логических операторов
        expression = expression.replace(' and ', ' & ').replace(' or ', ' | ')

        return expression

# -----------------------------
# ASTConditionCompiler: преобразование if(...) в pd.eval / np.where
# -----------------------------
class ASTConditionCompiler:
    def __init__(self, context: ExpressionContext):
        self.ctx = context

    def compile_if(self, node_id: str, prop: str, expression: str) -> pd.Series:
        conditions = expression.split('if(')
        conditions = [c.rstrip(')') for c in conditions if c]
        conditions = list(reversed(conditions)) if len(conditions) > 1 else conditions

        result_series = None

        for idx, condition in enumerate(conditions):
            try:
                cond_expr, if_true, if_false = map(str.strip, condition.split(','))
                cond_expr = cond_expr.replace('=', '==') if '=' in cond_expr and '==' not in cond_expr else cond_expr
                cond_expr = ExpressionPreprocessor.preprocess(cond_expr)

                cond_result = pd.eval(cond_expr, engine='python', local_dict={'SWT': self.ctx.swt})
                if_true = self._resolve_value(node_id, if_true)
                if_false = result_series if result_series is not None else self._resolve_value(node_id, if_false)

                result_series = np.where(cond_result, if_true, if_false)

            except Exception as e:
                raise RuntimeError(f"Ошибка в условии '{condition}' ноды {node_id}.{prop}: {e}")

        return pd.Series(result_series)

    def _resolve_value(self, node_id: str, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return value

        try:
            return pd.eval(ExpressionPreprocessor.preprocess(value), engine='python', local_dict={'SWT': self.ctx.swt})
        except:
            return value

# -----------------------------
# ResultInserter: сохраняет результат в SWT[node.prop]
# -----------------------------
class ResultInserter:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def insert(self, node: str, prop: str, result: Any):
        col = f'{node}.{prop}'
        if isinstance(result, pd.Series):
            self.df[col] = pd.to_numeric(result, errors='coerce').fillna(0)
        else:
            self.df[col] = result

# -----------------------------
# ExpressionCalculator: orchestrator
# -----------------------------
@dataclass
class ExpressionCalculator:
    context: ExpressionContext

    def __call__(self, node_id: str, prop: str, expression: Any) -> None:
        swt = self.context.swt

        if isinstance(expression, (int, float, bool)):
            ResultInserter(swt).insert(node_id, prop, expression)
            return

        expression = ExpressionPreprocessor.preprocess(expression)

        if expression.startswith('if('):
            result = ASTConditionCompiler(self.context).compile_if(node_id, prop, expression)
        else:
            try:
                result = pd.eval(expression, engine='python', local_dict={'SWT': swt})
            except Exception as e:
                raise RuntimeError(f"Ошибка в выражении {node_id}.{prop}: {e}")

        ResultInserter(swt).insert(node_id, prop, result)

##GraphProcessor###
class Loader:
    def __init__(self, json_path: str):
        # Путь к файлу с графом в формате JSON
        self.json_path = json_path

    def load_graph_json(self) -> Dict[str, Any]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_swt_data(self, df: pd.DataFrame, swt_path: str) -> None:
        df.to_json(swt_path, orient='records', force_ascii=False, indent=2)

    def load_swt_data(self, swt_path: str) -> pd.DataFrame:
        return pd.read_json(swt_path, orient='records')
        
# -----------------------------
# 2. MetadataBuilder
# -----------------------------
# Отвечает за анализ структуры графа и подготовку данных для вычислений
class MetadataBuilder:
    def __init__(self, graph_data: Dict[str, Any]):
        # Сырые данные графа (узлы, ребра и т.д.)
        self.graph_data = graph_data
        self.node_index: Dict[str, int] = {}

    def build_node_index(self) -> Dict[str, int]:
        for idx, node in enumerate(self.graph_data.get('nodes', [])):
            self.node_index[node['id']] = idx
        return self.node_index

    def prepare_calculation_data(self) -> pd.DataFrame:
        """
        Собирает поля swt из каждого узла и возвращает DataFrame,
        пригодный для вычислений pandas.
        """
        records = [node.get('swt', {}) for node in self.graph_data.get('nodes', [])]
        return pd.DataFrame(records)


# -----------------------------
# 3. ExpressionManager
# -----------------------------
# Делегируем логику разбора и вычисления выражений через AST

class ExpressionParser:
    def parse(self, expr: str) -> ast.AST:
        """
        Преобразует строку выражения в синтаксическое дерево AST.
        Это позволяет безопасно модифицировать и проверять код перед выполнением.
        """
        return ast.parse(expr, mode='eval')

class RefTransformer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.Subscript:
        """
        Заменяем имена переменных вида "A" на SWT['A'],
        чтобы затем использовать наш DataFrame в eval.
        """
        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id='SWT', ctx=ast.Load()),
                slice=ast.Index(ast.Constant(node.id)),
                ctx=ast.Load()
            ),
            node
        )

class ExpressionEvaluator:
    def __init__(self, df: pd.DataFrame):
        # DataFrame, в котором хранится вся переменная SWT
        self.df = df

    def evaluate(self, expr: str) -> pd.Series:
        """
        1) Парсим строку в AST
        2) Трансформируем ссылки на переменные
        3) Компилируем и выполняем через eval
        Возвращаем pandas.Series с результатом для каждой строки.
        """
        tree = ExpressionParser().parse(expr)
        tree = RefTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        code = compile(tree, '<expr>', 'eval')
        return eval(code, {}, {'SWT': self.df})

# -----------------------------
# 4. MegaNodeProcessor
# -----------------------------
# Специализированная агрегация (меганода)
class MegaNodeProcessor:
    def __init__(
        self,
        df: pd.DataFrame,
        graph_name: str,
        periods: list,
        calculated_properties: Dict[str, bool],
        dict_nodes_properties: Dict[str, str],
        nodes: list,
        node_numbers: Dict[str, int],
        mega_node_path: str,
        calculate_expression_fn: callable
    ):
        self.df = df
        self.graph_name = graph_name
        self.periods = periods
        self.calculated_properties = calculated_properties
        self.dict_nodes_properties = dict_nodes_properties
        self.nodes = nodes
        self.node_numbers = node_numbers
        self.mega_node_path = mega_node_path
        self.calculate_expression = calculate_expression_fn

    def process_meganode(self) -> pd.DataFrame:
        # Инициализация
        mega_node_primitive_re = [
            r"Risk_[0-9]{1,5}\.current_risk_impact",
            r"Risk_[0-9]{1,5}\.current_risk_value"
        ]
        mega_node_base = pd.DataFrame(self.df['_t'])

        # Приведение форматов PeriodSetup
        for col in self.df.columns:
            if 'PeriodSetup' in col:
                self.df[col] = self.df[col].astype(str)

        for period in self.periods:
            self.df[period] = self.df[period].apply(lambda x: f"0{x}" if len(x) == 6 and x[0] != '0' else x)

        last_period_raw = self.df[self.periods[1]].iloc[0]
        last_period = (pd.to_datetime(last_period_raw, format='%m.%Y') - pd.DateOffset(months=1)).strftime('%m.%Y')

        self.df['mY'] = (pd.to_datetime(self.df['_t'], unit='s') + pd.DateOffset(hours=3)).dt.strftime('%m.%Y')

        # Загрузка существующей меганоды (если есть)
        mega_node_exists = Path(self.mega_node_path).exists()
        mega_node = pd.read_json(self.mega_node_path, orient='records', lines=True) if mega_node_exists else None

        # Основной цикл вычислений
        while not all(self.calculated_properties.values()):
            for cp, is_calculated in self.calculated_properties.items():
                if is_calculated:
                    continue
                try:
                    node_id, prop = cp.split('.')
                    self.calculate_expression(self.df, node_id, prop, self.dict_nodes_properties[cp])
                    self.calculated_properties[cp] = True
                    print(f'{cp} OK. Рассчитано {sum(self.calculated_properties.values())} из {len(self.calculated_properties)}')

                    # Добавление примитивов меганоды
                    for pattern in mega_node_primitive_re:
                        if re.search(pattern, cp):
                            mega_node_base = mega_node_base.merge(self.df[['_t', cp]], on='_t', how='left')
                            mega_node_base[f"{self.graph_name}.{prop}"] = mega_node_base[cp]
                            del mega_node_base[cp]

                    # Факторный анализ
                    if re.search(r'FactorAnalysis_[0-9]{1,5}\.value', cp):
                        expr = self.nodes[self.node_numbers[node_id]]['properties']['name']['expression'].replace('"', '')
                        mega_node_base = mega_node_base.merge(self.df[['_t', cp]], on='_t', how='left')
                        mega_node_base[f"{self.graph_name}.{expr}"] = mega_node_base[cp]
                        del mega_node_base[cp]

                except KeyError:
                    continue
                except TypeError as te:
                    raise TypeError(f"Ошибка: {str(te)}")
                except NameError as ne:
                    msg = (
                        f"Порт (выражение) {self.dict_nodes_properties[cp]} не найден в свойстве {prop} "
                        f"примитива {node_id} ({self.nodes[self.node_numbers[node_id]]['properties']['name']['expression']}). "
                        f"Проверь заполнение свойства {cp}."
                    )
                    raise NameError(f"Ошибка: {msg}")
                except Exception as ex:
                    print(f"{str(ex)}\n{cp} ; {self.dict_nodes_properties[cp]}")
                    raise

        # Фильтрация по периоду
        swt_filtered = self.df[self.df['mY'] == last_period]

        # Объединение и сохранение меганоды
        if not mega_node_exists:
            mega_node = mega_node_base
        else:
            preserved = [col for col in mega_node_base.columns if col != '_t']
            mega_node = mega_node.drop(columns=preserved, errors='ignore')
            mega_node = mega_node.merge(mega_node_base, on='_t', how='right')

        mega_node = mega_node.reset_index(drop=True)
        mega_node.to_json(self.mega_node_path, orient='records', lines=True, index=False)
        return self.df

# -----------------------------
# 5. AlertProcessor
# -----------------------------
# Вычисление масок-алертов и отправка писем
class AlertProcessor:
    def __init__(self, df: pd.DataFrame, alert_config: Dict[str, Any]):
        self.df = df
        self.config = alert_config

    def compute_alerts(self) -> pd.DataFrame:
        """
        По конфигу пробегаем по выражениям, создаём булевы колонки.
        """
        for name, expr in self.config.get('conditions', {}).items():
            self.df[name] = ExpressionEvaluator(self.df).evaluate(expr)
        return self.df

    def send_email(self, to_addr: str, subject: str, body: str) -> None:
        """
        Отправка простого текстового письма через localhost SMTP.
        """
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['To'] = to_addr
        msg.set_content(body)
        with smtplib.SMTP('localhost') as s:
            s.send_message(msg)

# -----------------------------
# 6. GraphUploader
# -----------------------------
# HTTP-клиент для загрузки графа
class GraphUploader:
    def __init__(self, api_url: str, token: str):
        self.api_url = api_url
        self.token = token

    def upload_graph(self, graph_data: Dict[str, Any]) -> requests.Response:
        """
        POST JSON на endpoint с авторизацией.
        """
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        resp = requests.post(self.api_url, json=graph_data, headers=headers)
        resp.raise_for_status()
        return resp

# -----------------------------
# 7. Orchestrator
# -----------------------------
# Объединяет все шаги в единый workflow
class GraphProcessorOrchestrator:
    def __init__(
        self,
        loader: Loader,
        builder: MetadataBuilder,
        evaluator: ExpressionEvaluator,
        mega_proc: MegaNodeProcessor,
        alert_proc: AlertProcessor,
        uploader: GraphUploader
    ):
        self.loader = loader
        self.builder = builder
        self.evaluator = evaluator
        self.mega_proc = mega_proc
        self.alert_proc = alert_proc
        self.uploader = uploader

    def run(self) -> Any:
        # 1. Загрузка начального графа
        graph = self.loader.load_graph_json()
        # 2. Подготовка индексов и SWT-данных
        self.builder.graph_data = graph
        node_index = self.builder.build_node_index()
        swt_df = self.builder.prepare_calculation_data()

        # 3. Вычисление примера выражения
        swt_df['new_col'] = self.evaluator.evaluate('A + B * C')

        # 4. Обработка меганоды
        swt_df = self.mega_proc.process_meganode()
        # 5. Вычисление алертов
        swt_df = self.alert_proc.compute_alerts()

        # 6. Сохранение промежуточных результатов
        self.loader.save_swt_data(swt_df, 'swt_out.json')

        # 7. Обновление графа и загрузка на сервер
        graph['swt'] = swt_df.to_dict(orient='records')
        result = self.uploader.upload_graph(graph)
        return result.json()


def _process_alerts(self, SWT):
    alert_nodes = [node for node in self.graph.nodes.values() if 'Alert' in node.primitive_id]
    
    if not alert_nodes:
        return
    
    for alert_node in alert_nodes:
        node_id = alert_node.primitive_id
        
        conditions = {}
        for prop_name, prop in alert_node.properties.items():
            if re.match(r'condition_\d+', prop_name):
                conditions[prop_name] = prop.expression
        
        for cond_name, expr in conditions.items():
            try:
                # Заменяем ссылки на свойства других узлов
                for other_node in self.graph.nodes.values():
                    for other_prop in other_node.properties:
                        prop_key = f"{other_node.primitive_id}.{other_prop}"
                        if prop_key in expr:
                            expr = expr.replace(prop_key, f"SWT['{prop_key}']")
                
                # Вычисляем условие
                SWT = ExpressionCalculator.calculate_expression(
                    SWT, 
                    node_id, 
                    cond_name, 
                    expr, 
                    self.graph
                )
            except Exception as e:
                raise RuntimeError(
                    f"Ошибка вычисления условия {cond_name} "
                    f"в алерте {node_id}: {str(e)}"
                )
        
        if 'root_condition' in alert_node.properties:
            root_expr = alert_node.properties['root_condition'].expression
            try:
                # Дополнительная проверка и замена
                if not root_expr:
                    print(f"Предупреждение: root_condition пустой в алерте {node_id}")
                    continue
                
                # Заменяем ссылки на свойства
                for prop_name in alert_node.properties:
                    if prop_name != 'root_condition' and prop_name in root_expr:
                        root_expr = root_expr.replace(
                            prop_name, 
                            f"SWT['{node_id}.{prop_name}']"
                        )
                
                # Вычисление
                SWT = ExpressionCalculator.calculate_expression(
                    SWT, 
                    node_id, 
                    'root_condition', 
                    root_expr, 
                    self.graph
                )
            except Exception as e:
                print(
                    f"Предупреждение: не удалось вычислить root_condition "
                    f"для алерта {node_id}: {str(e)}"
                )
        
        SWT_filtered = SWT[SWT['mY'] == self.report_month_period]
        if SWT_filtered.empty:
            print(f"Нет данных за отчетный период {self.report_month_period} для алерта {node_id}")
            continue
        
        for prop_name in alert_node.properties:
            prop_key = f"{node_id}.{prop_name}"
            if prop_key in SWT_filtered.columns:
                value = SWT_filtered[prop_key].iloc[0]
                
                # Преобразование числовых значений
                if isinstance(value, (float, np.float64)):
                    value = round(value, 5)
                elif isinstance(value, np.int64):
                    value = int(value)
                
                alert_node.properties[prop_name].value = value
        
        alert_subject = alert_node.properties.get('alert_subject', GraphProperty('', '')).value or ''
        alert_text = alert_node.properties.get('alert_text', GraphProperty('', '')).value or ''
        email_address = alert_node.properties.get('email_address', GraphProperty('', '')).value or ''
        
        # Добавление информации о периоде
        month_num, year_num = self.report_month_period.split('.')
        month_name = self.MONTHES[int(month_num)-1]
        period_text = f"{month_name} {year_num} г."
        full_subject = f"{alert_subject} за {period_text}"
        
        alert_data = {
            'subject': full_subject,
            'text': alert_text,
            'email': email_address
        }
        
        alert_file_path = os.path.join('/opt', 'otp', 'external_data', 'alerts', f'{self.graph_name}_alert.json')
        send_alert = True
        
        # Проверяем предыдущее состояние алерта
        if os.path.exists(alert_file_path):
            try:
                with open(alert_file_path, 'r') as f:
                    prev_alert_data = json.load(f)
                
                # Сравниваем с текущим состоянием
                if (prev_alert_data.get('subject') == full_subject and
                    prev_alert_data.get('text') == alert_text and
                    prev_alert_data.get('email') == email_address):
                    send_alert = False
            except:
                pass
        
        if send_alert:
            try:
                # Сохраняем текущее состояние
                os.makedirs(os.path.dirname(alert_file_path), exist_ok=True)
                with open(alert_file_path, 'w') as f:
                    json.dump(alert_data, f)
                
                # Отправляем email
                sendAlert(
                    subject=full_subject,
                    message=alert_text,
                    email_to=email_address,
                    graph_name=self.graph_name
                )
                print(f"Алерт для графа {self.graph_name} отправлен на {email_address}")
            except Exception as e:
                print(f"Ошибка отправки алерта: {str(e)}")
    
    def _update_graph_values(self, SWT):
        # Обновляем значения свойств всех узлов
        for node in self.graph.nodes.values():
            for prop_name, prop in node.properties.items():
                prop_key = f"{node.primitive_id}.{prop_name}"
                if prop_key in SWT.columns:
                    # Получаем значение из DataFrame
                    value = SWT[prop_key].iloc[0]
                    
                    # Обработка числовых значений
                    if isinstance(value, (float, np.float64)):
                        value = round(value, 5)
                    
                    # Обновляем свойство
                    prop.value = value
                    prop.status = 'complete'
    
    def _upload_graph(self):
        # Конфигурация подключения
        URL, PORT, USERNAME, PASSWORD = '127.0.0.1', '6081', 'admin', 'admin'
        session = requests.Session()
        
        # Аутентификация
        login_url = f'http://{URL}:{PORT}/auth/login'
        response = session.post(
            login_url, 
            data={"login": USERNAME, "password": PASSWORD}
        )
        
        if response.status_code != 200:
            raise ConnectionError(f"Ошибка аутентификации: {response.status_code}")
        
        # Поиск ID графа
        fragments_url = f'http://{URL}:{PORT}/supergraph/v1/fragments'
        response = session.get(fragments_url)
        
        if response.status_code != 200:
            raise ConnectionError(f"Ошибка получения списка графов: {response.status_code}")
        
        graph_id = None
        for fragment in response.json()['fragments']:
            if fragment['name'] == self.graph_name:
                graph_id = fragment['id']
                break
        
        if not graph_id:
            raise ValueError(f"Граф {self.graph_name} не найден в системе")
        
        # Подготовка данных для отправки
        graph_data = self.graph.to_json()
        graph_url = f'http://{URL}:{PORT}/supergraph/v1/fragments/{graph_id}/graph'
        headers = {'Content-type': 'application/json;charset=UTF-8'}
        
        # Отправка данных
        response = session.put(
            graph_url, 
            data=json.dumps(graph_data), 
            headers=headers
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Ошибка обновления графа {self.graph_name}: "
                f"{response.status_code} - {response.text}"
            )
        
        # Дополнительное обновление метаданных
        meta_url = f'http://{URL}:{PORT}/supergraph/v1/fragments/{graph_id}'
        response = session.put(
            meta_url, 
            data=json.dumps({"name": self.graph_name})
        )
        
        if response.status_code != 200:
            print(f"Предупреждение: не удалось обновить метаданные графа")
    
    def calculate(self):
        if not fillDataLake(self.graph_name):
            raise RuntimeError(f'Ошибка заполнения DataLake для графа {self.graph_name}')
        
        self._prepare_calculation_data()
        
        SWT = self._load_swt_data()
        
        SWT['mY'] = (pd.to_datetime(SWT['_t'], unit='s') + pd.DateOffset(hours=3)).dt.strftime('%m.%Y')
        
        calculated_properties = {k: False for k in self.dict_nodes_properties.keys()}
        last_period = SWT[SWT.columns[SWT.columns.str.contains('PeriodSetup')][0]].iloc[0]
        
        while not all(calculated_properties.values()):
            for prop_key in list(calculated_properties.keys()):
                if not calculated_properties[prop_key]:
                    try:
                        node_id, prop_name = prop_key.split('.')
                        expression = self.dict_nodes_properties[prop_key]
                        
                        # Вычисление выражения
                        SWT = ExpressionCalculator.calculate_expression(
                            SWT, 
                            node_id, 
                            prop_name, 
                            expression, 
                            self.graph
                        )
                        
                        calculated_properties[prop_key] = True
                        print(f"Вычислено: {prop_key}")
                        
                    except KeyError:
                        # Возможно, зависимость появится позже
                        continue
                    except Exception as e:
                        raise RuntimeError(f"Ошибка вычисления {prop_key}: {str(e)}")
        
        self._process_mega_node(SWT)
        
        last_period = (pd.to_datetime(last_period, format='%m.%Y') - pd.DateOffset(months=1)).strftime('%m.%Y')
        SWT_filtered = SWT[SWT['mY'] == last_period]

        for node in self.graph.nodes.values():
            for prop_name, prop in node.properties.items():
                prop_key = f"{node.primitive_id}.{prop_name}"
                
                if prop_key in SWT.columns and not SWT_filtered.empty:
                    # Получение значения
                    value = SWT_filtered[prop_key].iloc[0]
                    
                    # Обработка числовых значений
                    if isinstance(value, (float, np.float64)):
                        if abs(value) >= 100:
                            value = round(value, 2)
                        else:
                            value = round(value, 5)
                    elif isinstance(value, np.int64):
                        value = int(value)
                    
                    # Обновление свойства
                    prop.value = value
                    prop.status = 'complete'
        
        self._process_alerts(SWT)
        

        self._save_swt_data(SWT)
        self._upload_graph()
        
        print(f"Расчет графа {self.graph_name} успешно завершен")
        return self.graph.to_json()
        
    def showerror(errText):
        print(errText)
        exit(1)

    def calculateGraph(graphName, graph_json):
        try:
            graph_structure = GraphStructure(graph_json)
            
            processor = GraphProcessor(graphName, graph_structure)
            return processor.calculate()
        except Exception as e:
            # Обработка ошибок
            err_msg = f"Critical error in graph calculation: {str(e)}"
            print(err_msg)
            showerror(err_msg)
