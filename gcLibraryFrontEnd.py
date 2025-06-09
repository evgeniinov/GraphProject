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
        
class GraphStructure:

    def __init__(self, graph_json=None):
        self.nodes = {}
        self.connections = []
        
        if graph_json:
            self.load_from_json(graph_json)
    
    def load_from_json(self, graph_json):
        # Создаем узлы
        for node_data in graph_json['graph']['nodes']:
            node = GraphNode(
                node_data['primitiveID'], 
                node_data['primitiveName']
            )
            node.update_from_dict(node_data)
            self.nodes[node.primitive_id] = node
        
        # Создаем связи
        for edge_data in graph_json['graph']['edges']:
            connection = GraphConnection(
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
        

class ExpressionCalculator:

    @staticmethod
    def custom_ifnull(x, default_value):
        return default_value if pd.isnull(x) else x
        
    @staticmethod
    def replace_calc(match):
        calc_pattern = match.group(0)
        return calc_pattern if calc_pattern.startswith('SWT[') else f"SWT['{match.group(1)}']"
    
    @staticmethod
    def calculate_expression(SWT, node_id, property_name, expression, graph_structure):

        try:
            if isinstance(expression, str):
                if 'Calc' in expression:
                    expression = re.sub(r'(SWT\[.*?]|Calc(.*?)\.value)', replace_calc, expression)
                if 'ifnull' in expression:
                    expression = expression.replace(' ', '').replace('ifnull(', '')[:-3]
            elif (isinstance(expression, int) or isinstance(expression, bool) or isinstance(expression, float)):
                SWT[f'{node}.{property}'] = expression
                return SWT
            if isinstance(expression, str) and not 'if(' in expression:
                SWT[f'{node}.{property}'] = pd.eval(expression, engine='python', local_dict=SWT)
                if not isinstance(SWT[f'{node}.{property}'], int):
                    SWT[f'{node}.{property}'] = SWT[f'{node}.{property}']
                SWT[f'{node}.{property}'] =  SWT[f'{node}.{property}'].fillna(value=0)
                return SWT
            elif not 'if(' in expression:
                SWT[f'{node}.{property}'] = expression
                return SWT
                
            if 'if(' in expression:
                if f'{node}.{property}' in SWT:
                    input(f'{node}.{property} уже присутствует!')
                    exit(1)
                    return SWT
                else:
                    print(f'Вычисления условий ноды {node}; свойства {property} выражения {expression}')
                if re.search(' {1,10}and {1,10}', expression):
                    rem= re.search(' {1,10}and {1,10}', expression).group(0)
                    expression = expression.replace(rem, '&').replace(' ','')

                conditions = expression.split('if(')
                conditions = [c for c in conditions if c]

                if len(conditions)!=1:
                    conditions = list(reversed(conditions))

                newColumn = ''
                for condition in conditions:
                    if '")' in condition: 
                        cleanCondition = condition.replace(')','')
                    else:
                        cleanCondition = condition
                 
                    cond, ifTrue,ifFalse = cleanCondition.split(',')
                    if re.search('[>=<]{1,2}', cond):
                        sign = re.search('[>=<]{1,2}', cond).group(0)
                        if sign == '=':
                            cond = cond.replace('=', '==')
                    if len(conditions)!=1: print(f'{cond} OK')

                    print(f'{cond} : ifTrue : {ifTrue}, ifFalse : {ifFalse}')
                    
                    thisNodeProperties = nodes[nodeNumbers[node]]['properties'].keys()
                    if ifTrue in thisNodeProperties:
                        ifTrue = nodes[nodeNumbers[node]]['properties']['ifTrue']['expression']
                    if ifFalse in thisNodeProperties:
                        ifFalse = nodes[nodeNumbers[node]]['properties']['ifFalse']['expression']          
                    #повторный анализ на наличие SWT в условии
                    try:
                        primitivesC = [ppt for ppt in nodes[nodeNumbers[node]]['properties'].keys() if ppt!=property]
                        for pc in primitivesC:
                            seps = r'[><=!]='
                            additionalCheckPropCond = re.split(seps, cond, maxsplit=1)[0].replace(' ','')
                            if pc in cond and not (f"SWT[" in cond or  f'SWT["' in cond) and pc == additionalCheckPropCond:
                                cond = cond.replace(pc, f"SWT['{node}.{pc}']")
                                                  
                        for itf in [ifTrue, ifFalse]:
                            if nodes[nodeNumbers[node]]['properties'][pc]['expression']:
                                if pc in itf and not f"SWT['{node}.{pc}']" in itf and not re.search('[in|out]Port', nodes[nodeNumbers[node]]['properties'][pc]['expression']):
                                    if itf == ifTrue:
                                        ifTrue = nodes[nodeNumbers[node]]['properties'][pc]['expression']
                                    elif itf == ifFalse:
                                        ifFalse = nodes[nodeNumbers[node]]['properties'][pc]['expression']
                                if len([p for p in primitivesC if p in ifTrue])>0 and isinstance(ifTrue, str):
                                    print(f'2.221B : {ifTrue}')
                                    if not 'SWT[' in ifTrue:
                                        propertiesToReplace = [p for p in primitivesC if p in ifTrue]
                                        for ptr in propertiesToReplace:
                                            ifTrue = ifTrue.replace(ptr, f"SWT['{node}.{ptr}']") if f'{node}.{ptr}' in SWT.columns else ifTrue
                                    print('2.221A')
                                if len([p for p in primitivesC if p in ifFalse])>0 and isinstance(ifFalse,str):
                                    if not 'SWT[' in ifFalse:
                                        ifFalse = ifFalse.replace(p, "SWT['{node}.{p}']")
                                    print('2.222')
                                print('2.22')               
                    except NameError as ne:
                        raise(ne)
                    except Exception as e:
                        raise(e)
                    
                    #if not newColumn:
                    try:
                        print(f'Вычисление {cond}')
                        result = pd.eval(cond, engine='python')
                        print(f'{cond} вычислено')                 
                    except TypeError as te:
                        errText = str(te)
                        errText = f'Проверьте корректность заполнения источников {node}.{property}, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}'
                        print(errText)
                        raise te(errText)
                    except KeyError as ke:
                        if 'root_condition' in property:
                            print(expression)
                        raise(ke)
                    except:
                        seps = r'[><=!]='
                        additionalCheckPropCond = re.split(seps, cond, maxsplit=1)[0].replace(' ','')
                        etext = f'Необходимо проверить корректность заполнения условного выражения {node}.{property}. Возможно, свойство {additionalCheckPropCond} отсутствует в примитиве..'
                        print(etext)
                        raise Exception(etext)
                    SWT[f'condition_element_{conditions.index(condition)}'] = result
                    conditionColumn = f'condition_element_{conditions.index(condition)}'

                    if isinstance(ifFalse, str):
                        ifFalse = ifFalse.replace(')','')
                    if not isinstance(ifFalse, int) and not not isinstance(ifFalse, float):
                        try:
                            ifFalse = float(ifFalse)
                        except:
                            ifFalse = ifFalse
                    
                    if conditions.index(condition) != 0:
                        ifFalse =  SWT[f'{node}.{property}']

                    print(ifTrue in nodes[nodeNumbers[node]]['properties'].keys())
                    print(f'2.4 : {ifTrue} . {ifFalse}')
                    
                    if isinstance(ifTrue, str):
                        ifTrueTrimmed = ifTrue.replace(' ', '')
                        if ifTrueTrimmed in nodes[nodeNumbers[node]]['properties'].keys():
                            ifTrue = nodes[nodeNumbers[node]]['properties'][ifTrueTrimmed]['expression']
                            
                    print('Переход к вычислению SWT condition')
                    try:
                        if not (isinstance(ifTrue, str) or isinstance(ifFalse, str)):
                            if ('inPort' in ifTrue or 'inPort' in ifFalse) or len([prop for prop in nodes[nodeNumbers[node]]['properties'].keys() if (prop in inTrue or prop in ifFalse)])>0:
                                SWT[f'{node}.{property}'] = pd.eval(ifTrue).where(SWT[conditionColumn], ifFalse)
                            else:
                                SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], ifTrue,ifFalse)
                        elif (isinstance(ifTrue, str)):
                            if re.search('\+|-|\*|/',ifTrue):
                                SWT[f'{node}.{property}'] = pd.eval(ifTrue).where(SWT[conditionColumn], ifFalse)
                            elif 'SWT' in ifTrue:
                                SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], pd.eval(ifTrue), pd.eval(ifFalse))
                            elif 'SWT' in ifFalse:
                                SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], pd.eval(ifTrue), pd.eval(ifFalse))
                            else:
                                SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], ifTrue,ifFalse)
                        else:
                            SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], ifTrue,ifFalse)
                        try:
                            SWT[f'{node}.{property}'] = pd.to_numeric(SWT[f'{node}.{property}'])
                        except ValueError:
                            pass
                        SWT.drop(columns=[conditionColumn])

                    except KeyError as ke:
                        if property == 'root_condition' : print(ifTrue)
                        print(f'Нет источников для {node}.{property}, продолжаю')
                        raise(ke)
                    except TypeError as te:
                        print(te)
                        input('!!!')                
                    except Exception as e:
                        print('ошибка вычисления')
                        print(pd.eval(ifTrue))
                        raise(e)
                        
            if len(conditions) > 1: print(SWT[f'{node}.{property}'])
            return SWT
        except ValueError as ve:
            if not nodes[nodeNumbers[node]]["properties"][property]["expression"]:
                veText = f'Пустое выражение ноды {node} свойства {property}, то есть ноды с именем {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}'
                raise ValueError(veText)
                if not 'Alert' in nodes[nodeNumbers[node]]['primitiveID']:
                    exit(1)
                else:
                    print('Условие не учтено! Рекомендуется дополнить условия для корректного расчёта алертов.')
            else:
                print(ve)
                pass
        except AttributeError as ae:
            input(ae)
            raise ae
            print(f'Проверьте заполнение источников {node}.{property} в вайде, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}')
            exit(1)
        except TypeError as te:
            print(expression)
            input(te)
            
            teText = str(te)
            if 'Вероятно' not in teText:
                errText = f'Проверьте корректность заполнения источников {node}.{property}, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}'
            else:
                errText = teText
            teText = f'Проверьте корректность заполнения источников {node}.{property}, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}'
            raise TypeError(teText)                              
        except SyntaxError as se:
            errText = str(se)
            print(f'ошибка {errText} ; нода {node} свойство {property} выражение {expression}')
            print(f'Также {dictNodeProperties}')
            exit(1)
        except Exception as e:
            raise(e)

class GraphProcessor:

    MEGA_NODE_PATH = os.path.join('/opt', 'otp', 'external_data', 'megaNode', 'megaNode.json')
    MONTHES = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 
               'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']
    
    def __init__(self, graph_name, graph_structure):
        self.graph_name = graph_name
        self.graph = graph_structure
        self.node_index = self._build_node_index()
        self.report_month_period = self._get_report_month_period()
        
    def _get_report_month_period(self):
        today_day = datetime.now().date().day
        report_date = (datetime.now() - relativedelta(months=2)) if today_day <= 20 else (datetime.now() - relativedelta(months=1))
        return report_date.strftime("%m.%Y")
    
    def _build_node_index(self):
        """Создает индекс для быстрого поиска узлов по ID"""
        return {node_id: idx for idx, node_id in enumerate(self.graph.nodes)}
    
    def _prepare_calculation_data(self):

        self.dict_nodes_properties = {}
        self.port_expressions = {}
        self.value_nodes = {}
        
        # выражения для портов
        for node in self.graph.nodes.values():
            if hasattr(node, 'init_ports') and node.init_ports:
                for port in node.init_ports:
                    if 'out' in port['primitiveName']:
                        port_key = f"{node.primitiveID}.{port['primitiveName']}"
                        self.port_expressions[port_key] = port['properties']['status']['expression']
        
        #виды прмитимов
        for node in self.graph.nodes.values():
            node_id = node.primitiveID
            self.value_nodes[node_id] = {}
            
            # Для узлов с прямым значением
            if 'value' in node.properties:
                self.value_nodes[node_id]['value'] = node.properties['value'].expression
            
            # Для KIR-узлов
            elif 'KIR' in node.primitive_name:
                for prop in ['budget_value', 'fact_value', 'boundary_value', 'critical_value']:
                    if prop in node.properties:
                        self.value_nodes[node_id][prop] = node.properties[prop].expression
            
            # Для Risk-узлов
            elif node_id.startswith('Risk_'):
                for prop in ['current_risk_impact', 'current_risk_value']:
                    if prop in node.properties:
                        self.value_nodes[node_id][prop] = node.properties[prop].expression
                        
            if node_id.startswith('Risk_'):
                for prop in ['current_risk_impact', 'current_risk_value']:
                    if prop in node.properties:
                        self.value_nodes[prop] = node.properties[prop].expression

            elif 'RiskFactor' in node_id:
            
                self.value_nodes[node_id]['value'] = node.properties['value'].expression
                
                for prop in ['expected_effect', 'value_to_goal']:
                    if prop in node.properties:
                        self.value_nodes[prop] = node.properties[prop].expression                  
            
            elif 'RiskMeas' in node_id:
                for prop in ['residual_risk_impact', 'residual_risk_value']:
                    if prop in node.properties:
                        self.value_nodes[node_id][prop] = node.properties[prop].expression

            elif 'FactorAnalysis_' in node_id:
                for prop in ['management_action', 'external_influence']:
                    if prop in node.properties:
                        self.value_nodes[node_id][prop] = node.properties[prop].expression

            elif 'Measures_' in node_id:
                msListNum=0
                for np, port in enumerate(node['initPorts']):
                    if 'inPort' in port['primitiveName']:
                        msListNum+=1
                if msListNum > 0:
                    for prop in ['value', 'efficiency']:
                        if prop in node.properties:
                            self.value_nodes[node_id][prop] = node.properties[prop].expression

            if node_id.startswith('RiskAppetiteGoal'):               
                for prop in ['goal_budget_value', 'goal_fact_value', 'risk_appetite_value', 'risk_appetite_exceeded']:
                    self.value_nodes[node_id][prop] = node.properties[prop].expression
                    
            if node['primitiveID'].startswith('RiskAppetiteRisk'):
                valueNodes[node['primitiveID']] = {}
                for prop in ['risk_appetite_impact', 'risk_appetite_value','risk_appetite_exceeded']:
                    valueNodes[node['primitiveID']][prop] = node['properties'][prop]['expression']
                                

        dependencies = {}
        
        # Сбор зависимостей из связей
        for conn in self.graph.connections:
            target_node = conn.target_node
            if target_node not in dependencies:
                dependencies[target_node] = [{}, {}]
            
            source_node = conn.source_node
            port_name = conn.target_port.split('_')[2]
            dependencies[target_node][0][source_node] = port_name
            dependencies[target_node][1] = self.value_nodes.get(target_node, {})
        
        # Формирование итоговых выражений
        for node_id, (sources, properties) in dependencies.items():
            for prop_name, expr in properties.items():
                current_expr = str(expr)
                
                # Замена ссылок на порты
                for source_node, port_name in sources.items():
                    if port_name in current_expr:
                        # Поиск выражения для порта
                        port_expr = None
                        for conn in self.graph.connections:
                            if (conn.target_node == node_id and 
                                conn.target_port == f"{node_id}_{port_name}"):
                                source_port = conn.source_port.replace('_out', '.out')
                                port_expr = self.port_expressions.get(source_port)
                                break
                        
                        if port_expr:
                            current_expr = current_expr.replace(
                                port_name, 
                                f"SWT['{source_node}.{port_expr}']"
                            )
                
                # Сохранение итогового выражения
                self.dict_nodes_properties[f"{node_id}.{prop_name}"] = current_expr
        
        # Шаг 4: Дополнительная обработка выражений
        for prop_key, expr in self.dict_nodes_properties.items():
            node_id, prop_name = prop_key.split('.')
            node = self.graph.nodes[node_id]
            
            # Замена ссылок на свойства внутри того же узла
            for other_prop in node.properties:
                if other_prop != prop_name and other_prop in expr:
                    pattern = rf"\b{re.escape(other_prop)}\b"
                    if not re.search(rf"SWT\['.*?\.{re.escape(other_prop)}'\]", expr):
                        expr = re.sub(
                            pattern, 
                            f"SWT['{node_id}.{other_prop}']", 
                            expr
                        )
            self.dict_nodes_properties[prop_key] = expr
    
    def _load_swt_data(self):
        swt_path = f'/opt/otp/external_data/SWT/{self.graph_name}/{self.graph_name}.json'
        return pd.read_json(swt_path, orient='records', lines=True)
    
    def _save_swt_data(self, SWT):
        swt_path = f'/opt/otp/external_data/SWT/{self.graph_name}/{self.graph_name}.json'
        SWT.to_json(swt_path, orient='records', lines=True)
    
    def _process_mega_node(self, SWT):
        ###МЕГАНода###
        megaNodePrimitivesRE = ['Risk_[0-9]{1,5}.current_risk_impact', 'Risk_[0-9]{1,5}.current_risk_value' ]
        megaNode0  = pd.DataFrame(SWT['_t'])
        ##############

        for col in SWT.columns:
            if 'PeriodSetup' in col: SWT[col] = SWT[col].astype(str)

        for period in periods:
            SWT[period] = SWT[period].apply(lambda x: f"0{x}" if len(x) == 6 and x[0] != '0' else x)

        lastPeriodP = SWT[periods[1]][0]
        #со смещением в 1 месяц
        lastPeriod = (pd.to_datetime(lastPeriodP, format='%m.%Y') - pd.DateOffset(months=1)).strftime('%m.%Y')
        SWT['mY'] = (pd.to_datetime(SWT['_t'], unit='s') + pd.DateOffset(hours=3)).dt.strftime('%m.%Y')
        rvalues = []
        #главная часть
        try:
            megaNode = pd.read_json(megaNodePath, orient='records', lines=True)
            megaNodeExists = True
        except FileNotFoundError:
            megaNodeExists = False
                
         
        while False in calculatedProperties.values():
            for cp in calculatedProperties.keys():
                try:
                    if not calculatedProperties[cp]:
                        calculateExpression(SWT, cp.split('.')[0], cp.split('.')[1], dictNodesProperties[cp])
                        calculatedProperties[cp] = True
                        sumTrue = sum(1 for value in calculatedProperties.values() if value)
                        print(f'{cp} OK. Рассчитано {sumTrue} свойств из {len(calculatedProperties)}')
                        
                        ###Проверка на megaNode###
                        for res in megaNodePrimitivesRE:
                            if re.search(res, cp):
                                left_side = graphName
                                megaNode0 = megaNode0.merge(SWT[['_t', cp]], how='left', on = '_t')
                                megaNode0[f"{graphName}.{cp.split('.')[1]}"] = megaNode0[cp]
                                del megaNode0[cp]
                        #ФА
                        if re.search('FactorAnalysis_[0-9]{1,5}.value', cp):
                            nameP = cp.split('.')[0]
                            rightSide = nodes[nodeNumbers[nameP]]['properties']['name']['expression'].replace('"','')
                            megaNode0 = megaNode0.merge(SWT[['_t', cp]], how='left', on = '_t')
                            megaNode0[f"{graphName}.{rightSide}"] = megaNode0[cp]
                            del megaNode0[cp]
                except KeyError as e:
                    continue
                    
                except TypeError as te:
                    raise TypeError(f'Ошибка: {str(te)}')
                
                except NameError as ne:
                    errText = str(ne)
                    ne = f'Порт (выражение) {dictNodesProperties[cp]} не найден в свойстве {cp.split(".")[1]} примитива {cp.split(".")[0]} ({nodes[nodeNumbers[cp.split(".")[0]]]["properties"]["name"]["expression"]}). Требуется проверить заполнение свойства {cp}'
                    raise NameError(f'Ошибка: {str(ne)}')
     
                except Exception as unhandled:
                    errText = str(unhandled)
                    print(errText)
                    print(f'{cp} ; {dictNodesProperties[cp]}')
                    raise
                    
        SWTFiltered = SWT[SWT['mY'] == lastPeriod]

        if not megaNodeExists:
            megaNode = megaNode0
        else:
            exColumns = [col for col in megaNode0.columns if col!='_t']
            megaNode = megaNode.drop(exColumns, axis=1, errors = 'ignore')
            megaNode = pd.merge(megaNode, megaNode0, how='right', on = ['_t'])
            megaNode = megaNode.reset_index(drop=True)
            
        megaNode.to_json(megaNodePath, orient='records', lines = True, index = False)    
        return 
    
    def _process_alerts(self, SWT):
        """Обрабатывает алерты"""
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
