import pandas as pd, numpy as np, re, os, requests
import json
from autoDataLakeL2 import fillDataLake
from sendEmailL import sendAlert
import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta

warnings.simplefilter("ignore")
megaNodePath = os.path.join('/opt', 'otp', 'external_data', 'megaNode','megaNode.json')
#Определение отчетного месяца#
todayDay = datetime.now().date().day
if todayDay<=20:
    reportMonthDate = (datetime.now() - relativedelta(months=2))
else:
    reportMonthDate = (datetime.now() - relativedelta(months=1))
#input(reportMonthDate)
reportMonthPeriod = reportMonthDate.strftime("%m.%Y")
#input(reportMonthPeriod)
#Конец определения отчетного месяца#

monthes = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']
errText = ''

def showerror(errText):
    print(errText)
    exit(1)
    
def custom_ifnull(x, default_value):
    return default_value if pd.isnull(x) else x

def replace_calc(match):
    calc_pattern = match.group(0)
    if calc_pattern.startswith('SWT['):
        return calc_pattern
    else:
        return f"SWT['{match.group(1)}']"

def calculateExpression(SWT, node, property, expression, ordered = False):
    try:
        #if f'{node}.{property}' not in SWT.columns: для повторного использования нужна перезапись
        if isinstance(expression, str):
            if 'Calc' in expression:
                expression = re.sub(r'(SWT\[.*?]|Calc(.*?)\.value)', replace_calc, expression)
            if 'ifnull' in expression:
                expression = expression.replace(' ', '').replace('ifnull(', '')[:-3]
        elif (isinstance(expression, int) or isinstance(expression, bool) or isinstance(expression, float)):
            #input(expression)
            SWT[f'{node}.{property}'] = expression
            return SWT
        #if 'KIR2' in node and property=='budget_value':
        #    input(f'{expresFsion}' )
        if isinstance(expression, str) and not 'if(' in expression:
            
            SWT[f'{node}.{property}'] = pd.eval(expression, engine='python', local_dict=SWT)
            if not isinstance(SWT[f'{node}.{property}'], int):
                #SWT[f'{node}.{property}'] = round(SWT[f'{node}.{property}'], 2)
                SWT[f'{node}.{property}'] = SWT[f'{node}.{property}']
            SWT[f'{node}.{property}'] =  SWT[f'{node}.{property}'].fillna(value=0)
            return SWT
        elif not 'if(' in expression:
            #input(expression)
            SWT[f'{node}.{property}'] = expression
            return SWT


        if 'if(' in expression:
            if f'{node}.{property}' in SWT:
                input(f'{node}.{property} уже присутствует!')
                exit(1)
                return SWT
            else:
                print(f'Вычисления условий ноды {node}; свойства {property} выражения {expression}')
            #input(expression)
            if re.search(' {1,10}and {1,10}', expression):
                rem= re.search(' {1,10}and {1,10}', expression).group(0)
                expression = expression.replace(rem, '&').replace(' ','')

            conditions = expression.split('if(')
            print(f'thisExpression: {expression}')
            conditions = [c for c in conditions if c]

            if len(conditions)!=1:
                conditions = list(reversed(conditions))
                #input(f'rev2 : {conditions}')

            newColumn = ''
            #input(conditions)
            for condition in conditions:
                #cleanCondition = condition.replace(')','').replace('(','')
                if '")' in condition: 
                    cleanCondition = condition.replace(')','')
                else:
                    cleanCondition = condition

                #input(cleanCondition)                    
                cond, ifTrue,ifFalse = cleanCondition.split(',')
                #input(cond)
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
                    #print(1)
                    primitivesC = [ppt for ppt in nodes[nodeNumbers[node]]['properties'].keys() if ppt!=property]
                    
                    print(2)
                    for pc in primitivesC:
                        seps = r'[><=!]='
                        additionalCheckPropCond = re.split(seps, cond, maxsplit=1)[0].replace(' ','')
                        #if 'condition_1' in cond: input(additionalCheckPropCond)
                        #print(f"2.1 : {pc} - {cond}")
                        #if pc in cond and (not f"SWT['{node}.{pc}']" in cond or not f'SWT["{node}.{pc}"]' in cond and not f'{node}.{pc}' in calculatedNodes.keys())\
                        if pc in cond and not (f"SWT[" in cond or  f'SWT["' in cond) and pc == additionalCheckPropCond:
                            cond = cond.replace(pc, f"SWT['{node}.{pc}']")
                            
                    
                    for itf in [ifTrue, ifFalse]:
                        if nodes[nodeNumbers[node]]['properties'][pc]['expression']:
                            print('2.21')
                            if pc in itf and not f"SWT['{node}.{pc}']" in itf and not re.search('[in|out]Port', nodes[nodeNumbers[node]]['properties'][pc]['expression']):
                                print('2.3')
                                if itf == ifTrue:
                                    ifTrue = nodes[nodeNumbers[node]]['properties'][pc]['expression']
                                elif itf == ifFalse:
                                    ifFalse = nodes[nodeNumbers[node]]['properties'][pc]['expression']
                            if len([p for p in primitivesC if p in ifTrue])>0 and isinstance(ifTrue, str):
                                print(f'2.221B : {ifTrue}')
                                if not 'SWT[' in ifTrue:
                                    propertiesToReplace = [p for p in primitivesC if p in ifTrue]
                                    #input(primitivesC)
                                    #input(propertiesToReplace)
                                    for ptr in propertiesToReplace:
                                        #ifTrue = ifTrue.replace(ptr, f"SWT['{node}.{ptr}']")
                                        ifTrue = ifTrue.replace(ptr, f"SWT['{node}.{ptr}']") if f'{node}.{ptr}' in SWT.columns else ifTrue
                                print('2.221A')
                            if len([p for p in primitivesC if p in ifFalse])>0 and isinstance(ifFalse,str):
                                if not 'SWT[' in ifFalse:
                                    ifFalse = ifFalse.replace(p, "SWT['{node}.{p}']")
                                print('2.222')
                            print('2.22')
                #print('2.5')                
                except NameError as ne:
                    #exit(1)
                    raise(ne)
                except Exception as e:
                    #print('ifs!')
                    #exit(1)
                    raise(e)
                #input(cond)
                print(cond)
                #if not newColumn:
                try:
                    print(f'Вычисление {cond}')
                    
                   
                    #if 'Calc_5967.value' in SWT.columns and 'Calc_5980.value' in SWT.columns:
                    #    input(SWT[['Calc_5967.value','Calc_5980.value']])
                    #print(2.2)
                    result = pd.eval(cond, engine='python')
                    print(f'{cond} вычислено')
                    
                    #if 'Data_21676.value' in cond:
                    #    input(result)
                    #if 'condition_1' in cond:
                    #    print(type(SWT[f'{node}.condition_1'].iloc[0]))
                        #input(f'нужное условие {cond}: {result}')                    
                except TypeError as te:
                    errText = str(te)
                    #print(te)
                    errText = f'Проверьте корректность заполнения источников {node}.{property}, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}'
                    print(errText)
                    raise te(errText)
                    #print(f'Проверьте корректность заполнения источников {node}.{property}, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}')
                    #print(f' {node} {property} {expression}')
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
                    #input(f'{cond} ; {sign}')
                SWT[f'condition_element_{conditions.index(condition)}'] = result
                conditionColumn = f'condition_element_{conditions.index(condition)}'
                #print('2.3')
                #if len(conditions)!=1: print(SWT[conditionColumn])
                #SWT[f'{node}.{property}'] = np.where(SWT[f'condition_element_{conditions.index(condition)}'], ifTrue, ifFalse)
                #если это число...
                #перенесено вправо с  этого места и  до конца секции
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
                #print(f'!!!!!!!!!{ifTrue == "notification_1"}')
                print(f'2.4 : {ifTrue} . {ifFalse}')
                
                if isinstance(ifTrue, str):
                    ifTrueTrimmed = ifTrue.replace(' ', '')
                    if ifTrueTrimmed in nodes[nodeNumbers[node]]['properties'].keys():
                        ifTrue = nodes[nodeNumbers[node]]['properties'][ifTrueTrimmed]['expression']
                        
                print('Переход к вычислению SWT condition')
                try:
                    print(f'iftrue:{ifTrue}, ifFalse : {ifFalse}')
                    if not (isinstance(ifTrue, str) or isinstance(ifFalse, str)):
                        print(1)
                        if ('inPort' in ifTrue or 'inPort' in ifFalse) or len([prop for prop in nodes[nodeNumbers[node]]['properties'].keys() if (prop in inTrue or prop in ifFalse)])>0:
                            print('swt01')
                            SWT[f'{node}.{property}'] = pd.eval(ifTrue).where(SWT[conditionColumn], ifFalse)
                            print(f' condition {cond} OK')
                        else:
                            print('swt02')
                            SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], ifTrue,ifFalse)
                    elif (isinstance(ifTrue, str)):
                        if re.search('\+|-|\*|/',ifTrue):
                            print('swt1')
                            SWT[f'{node}.{property}'] = pd.eval(ifTrue).where(SWT[conditionColumn], ifFalse)
                            print('20 1 OK')
                        elif 'SWT' in ifTrue:
                            print('swt2')
                            SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], pd.eval(ifTrue), pd.eval(ifFalse))
                            #                        and 'SWT' in ifFalse:
                            #input(f'{node}{property}: {ifTrue}')
                            #SWT[f'{node}.{property}'] = pd.eval(ifTrue).where(SWT[conditionColumn], ifFalse)
                            
                            #SWT[f'{node}.{property}'] = pd.eval(ifTrue).where(SWT[conditionColumn], pd.eval(ifFalse))
                            print('20 2T OK')
                        elif 'SWT' in ifFalse:
                            print('swt3')
                            SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], pd.eval(ifTrue), pd.eval(ifFalse))
                            #input(f'{node}{property}: {ifTrue}')
                            #if f'{node}.{property}' == 'FactorAnalysis_776.management_action':
                                #print(SWT[conditionColumn])
                                #print(pd.eval(ifFalse).where(SWT[conditionColumn], ifFalse))
                                #input(ifFalse)
                                
                        #    SWT[f'{node}.{property}'] = pd.eval(ifFalse)
                        #   #.where(SWT[conditionColumn], ifFalse)
                        #    print('20 2F OK')                            
                        else:
                            print('swtelse4')
                            SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], ifTrue,ifFalse)
                            print('20 3 OK')
                    else:
                        print(3)
                        SWT[f'{node}.{property}'] = np.where(SWT[conditionColumn], ifTrue,ifFalse)
                    try:
                        SWT[f'{node}.{property}'] = pd.to_numeric(SWT[f'{node}.{property}'])
                    except ValueError:
                        pass
                    SWT.drop(columns=[conditionColumn])
                    #if len(conditions) !=1:
                    #    print(f'{cond} {ifTrue}!!! {ifFalse}')
                    #    print(SWT[f'{node}.condition_1'])
                except KeyError as ke:
                    if property == 'root_condition' : print(ifTrue)
                    print(f'Нет источников для {node}.{property}, продолжаю')
                    #input('Ошибка!!!')
                    #raise(ke)
                    raise(ke)
                except TypeError as te:
                    print(te)
                    if 'Data_21676.value' in cond:
                        input(result)
                        
                    input('!!!')                
                except Exception as e:
                    print('ошибка вычисления')
                    print(pd.eval(ifTrue))
                    raise(e)
                    
                    #print(f'ifTrue {ifTrue} ; ifFalse {ifFalse}')
                    #input(SWT[f'condition_element_{conditions.index(condition)}'])
                    #pd.eval(ifTrue)
                    #input('!')
            
                   
            #input(SWT[f'{node}.{property}'])
        if len(conditions) > 1: print(SWT[f'{node}.{property}'])
        return SWT
    except ValueError as ve:
        #print(f'{node}; {property} {expression}')
        #raise
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
        #input(te)
        #print(nodes[nodeNumbers[node]]["properties"]["name"]["expression"])
        print(expression)
        #print(SWT['KIR2_280.budget_value'])
        input(te)
        
        teText = str(te)
        if 'Вероятно' not in teText:
            errText = f'Проверьте корректность заполнения источников {node}.{property}, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}'
        else:
            errText = teText
        teText = f'Проверьте корректность заполнения источников {node}.{property}, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}'
        #errText = f'Проверьте корректность заполнения источников {node}.{property}, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}'
        #print(errText)
        #input('!!!')
        raise TypeError(teText)                              
        #print(f'Проверьте корректность заполнения источников {node}.{property}, то есть ноды с именем  {nodes[nodeNumbers[node]]["properties"]["name"]["expression"]}')
        #print(f' {node} {property} {expression}')
        
        #print(SWT[['Data_19657.value', 'Data_19666.value']])
        #raise
        #exit(1)
    except SyntaxError as se:
        errText = str(se)
        print(f'ошибка {errText} ; нода {node} свойство {property} выражение {expression}')
        print(f'Также {dictNodeProperties}')
        exit(1)
    except Exception as e:
        
        #print(SWT['Calc2_5980.value']/1000000-SWT['Calc2_5967.value']/1000000)
        #input(SWT['Measures_459.value'])
        raise(e)
        #raise

def calculateGraph(graphName, graph):
    import numpy as np
    
    graphToProcess = graphName
    if fillDataLake(graphToProcess):
        print(f'DataLake графа {graphToProcess} успешно заполнено')
    else:
        showerror(f'Ошибка заполнения DataLake')

    URL,PORT,USERNAME,PASSWORD,session = '127.0.0.1','6081','admin','admin',requests.Session()
    url, body = f'http://{URL}:{PORT}/auth/login', {"login": USERNAME, "password": PASSWORD}
    r = session.post(url, data=body)
    if r.status_code != 200: showerror(f"Ошибка получения токена, код {r.status_code}")
    url = f'http://{URL}:{PORT}/supergraph/v1/fragments'
    r = session.get(url)
    if r.status_code != 200:
        showerror("Ошибка загрузки суперграфа, код {r.status_code}")
    errors = False
    requiredDictionary,thisJson,pathGDict  = {},{},{}
    path1, found = "", False
    #для упрощения список графов - 1 граф
    SWTPath = r'/opt/otp/external_data/SWT/' + f'{graphToProcess}//{graphToProcess}.json'

    for i in r.json()['fragments']:
        if i['name'] == graphToProcess:
            id = i['id']
            path1 = f'http://{URL}:{PORT}/supergraph/v1/fragments/{id}/graph'
            pathG = f'http://{URL}:{PORT}/supergraph/v1/fragments/{id}'
            pathGDict['name'] = i['name']
            break
            '''
            try:
                thisJsonR = session.get(path1)
                if thisJsonR.status_code == 200:
                    thisJson = thisJsonR.text
                    rp = ',"status":"success","status_code":200,"error":""'
                    thisJson = thisJson.replace(rp, '')
                    thisJson = thisJson.encode('utf-8')
                else:
                    print(f"Код ошибки {thisJsonR.status_code}")
                    exit()
            except:
                pass
            requiredDictionary['id'] = i['id']
            requiredDictionary['name'] = i['name']
            pathGDict['name'] = i['name']
            break
            '''

    headers = {'Accept' : "application/json, text/plain, */*", 'Content-type' : 'application/json;charset=UTF-8'}
    warningText = ''
    ##thisJsonS = thisJson.decode('utf-8')
    jsonGraph = graph
    #########thisJsonJSON['graph']

    global nodes
    global edges
    nodes = jsonGraph['graph']['nodes']
    edges = jsonGraph['graph']['edges']
    calculatedNodes = {}
    valueNodes = {}
    dictionaryNodes,dictionaryNodesStatus = {}, {}
    quickCalculate = ['Data', 'PeriodSetup']
    order2Calculate = ['Measures']
    periods = []
    global nodeNumbers
    nodeNumbers = {}
    alerted = False
    for n, node in enumerate(nodes):
        calculatedNodes[node['primitiveID']] = False if (not 'Data_' in node['primitiveID'] and not 'PeriodSetup' in node['primitiveID'] and not 'DataLakeNode' in node['primitiveID'] \
                                                         and not 'ExportNode' in node['primitiveID']) else True
        nodeNumbers[node['primitiveID']] = n

        if 'PeriodSetup' in node['primitiveID']:
            periods.append(f"{node['primitiveID']}.start")
            periods.append(f"{node['primitiveID']}.finish")
        if 'value' in node['properties']: #добавляем только те, которые не в списке
            valueNodes[node['primitiveID']] = {}
            valueNodes[node['primitiveID']]['value'] = node['properties']['value']['expression']
        elif 'KIR' in node['primitiveName']:
            valueNodes[node['primitiveID']] = {}
            valueNodes[node['primitiveID']]['budget_value'] = node['properties']['budget_value']['expression']
            valueNodes[node['primitiveID']]['fact_value'] = node['properties']['fact_value']['expression']
            valueNodes[node['primitiveID']]['boundary_value'] = node['properties']['boundary_value']['expression']
            valueNodes[node['primitiveID']]['critical_value'] = node['properties']['critical_value']['expression']

        if node['primitiveID'].startswith('Risk_'):
            valueNodes[node['primitiveID']] = {}
            valueNodes[node['primitiveID']]['current_risk_impact'] = node['properties']['current_risk_impact']['expression']
            valueNodes[node['primitiveID']]['current_risk_value'] = node['properties']['current_risk_value']['expression']
        elif 'RiskFactor' in node['primitiveID']:
            valueNodes[node['primitiveID']] = {}
            valueNodes[node['primitiveID']]['value'] = node['properties']['value']['expression']
            
            if node['properties']['expected_effect']['expression']:
                valueNodes[node['primitiveID']]['expected_effect'] = node['properties']['expected_effect']['expression']
                
            #valueNodes[node['primitiveID']]['value'] = node['properties']['value']['expression']
            if 'value_to_goal' in node['properties'].keys():
                valueNodes[node['primitiveID']]['value_to_goal'] = node['properties']['value_to_goal']['expression']
        elif 'RiskMeas' in node['primitiveID']:
            valueNodes[node['primitiveID']] = {}
            valueNodes[node['primitiveID']]['residual_risk_impact'] = node['properties']['residual_risk_impact']['expression']
            valueNodes[node['primitiveID']]['residual_risk_value'] = node['properties']['residual_risk_value']['expression']
        elif 'FactorAnalysis_' in node['primitiveID']:
            valueNodes[node['primitiveID']] = {}
            valueNodes[node['primitiveID']]['value'] = node['properties']['value']['expression']
            valueNodes[node['primitiveID']]['management_action'] = node['properties']['management_action']['expression']
            valueNodes[node['primitiveID']]['external_influence'] = node['properties']['external_influence']['expression']
        elif 'Measures_' in node['primitiveID']:
            msListNum=0
            for np, port in enumerate(node['initPorts']):
                if 'inPort' in port['primitiveName']:
                    msListNum+=1
            if msListNum > 0:
                valueNodes[node['primitiveID']] = {}
                valueNodes[node['primitiveID']]['value'] = node['properties']['value']['expression']
                valueNodes[node['primitiveID']]['efficiency'] = node['properties']['efficiency']['expression']

        if node['primitiveID'].startswith('RiskAppetiteGoal'):
            valueNodes[node['primitiveID']] = {}
            for p in ['goal_budget_value', 'goal_fact_value', 'risk_appetite_value', 'risk_appetite_exceeded']:
                valueNodes[node['primitiveID']][p] = node['properties'][p]['expression']
        if node['primitiveID'].startswith('RiskAppetiteRisk'):
            valueNodes[node['primitiveID']] = {}
            for p in ['risk_appetite_impact', 'risk_appetite_value','risk_appetite_exceeded']:
                valueNodes[node['primitiveID']][p] = node['properties'][p]['expression']

     ####special case
        if 'name' in node['properties'].keys() and 'if(' in node['properties']['name']['expression']:
            valueNodes[node['primitiveID']]['name'] = node['properties']['name']['expression']

        ####Alert part####
        if 'Alert' in node['primitiveID']:
            alerted = True
         
    #Делаем список списков (ДВА списка)
    #первый элемент списка - это ноды, которые подцепляются
    #второй элемент списка - это операции над inPort обычно
    #потом заменяем inPort на соответствующие ноды...
    for e, edge in enumerate(edges):
        thisEdge = edge['targetNode']
        if not thisEdge in dictionaryNodes:
            thisNode = thisEdge
            dictionaryNodes[thisEdge] = [{},{}]

        dictNode = dictionaryNodes[thisEdge][0] if dictionaryNodes[thisEdge] else {}

        dictNode[edge['sourceNode']] = edge['targetPort'].split('_')[2]
        dictionaryNodes[thisEdge][0] = dictNode
        valueRequired = valueNodes[edge['targetNode']]

        vNode = dictionaryNodes[thisEdge][1] if dictionaryNodes[thisEdge][1] else {}
        if not [key for key in valueRequired.keys() if key in vNode.keys()]:
            dictionaryNodes[thisEdge][1] = valueRequired

    #словарь примитивов и выражений портов для использования в дальнейшем
    #dictNodesProperties - как считается конкретно каждое свойство
    #portExpressions : выражения в ИСХОДЯЩИХ (out) портах примитивов
    dictNodesProperties, portExpressions = {},{}
    for n, node in enumerate(nodes):
        thisPID = node['primitiveID']
        if 'initPorts' in node.keys():
            for p, port in enumerate(node['initPorts']):
                primitiveName = port['primitiveName']
                if 'out' in primitiveName:
                    if not port['properties']['status']['expression']:
                        showerror(f"Порт {primitiveName} примитива {thisPID} ({node['properties']['name']['value']}) не заполнен")
                    portExpressions[f'{thisPID}.{primitiveName}'] = port['properties']['status']['expression']
                    if re.search('avg|sum',  port['properties']['status']['expression']):
                        showerror(f"Скорректируйте выражения исходящих портов примитива {thisPID}, то есть {node['properties']['name']['expression']}")

    for dtc in dictionaryNodes.keys():
        for pToC in dictionaryNodes[dtc][1].keys():
            currentValue = dictionaryNodes[dtc][1][pToC]
            for k1,v1 in dictionaryNodes[dtc][0].items():
                if isinstance(currentValue, int) or isinstance(currentValue, float) or isinstance(currentValue, bool):
                    raise TypeError(f'Значение {currentValue} свойства {pToC} РАСЧЁТНОГО примитива {dtc} ({nodes[nodeNumbers[dtc]]["properties"]["name"]["expression"]}) - число, хотя должно быть выражение. Проверьте заполнение свойства.')

                if v1 in currentValue:
                    pToFind = f'{dtc}_{v1}'
                    for e,edge in enumerate(edges):
                        if edge['targetPort'] == pToFind:
                            eToFind = edge['sourcePort']
                            try:
                                eToFind = eToFind.replace('_out','.out')
                                #input('replaced')
                            except:
                                input('error replaced, строка 512')
                            vToReplace = portExpressions[eToFind]
                            break
                    currentValue = currentValue.replace(v1, f"SWT['{k1}.{vToReplace}']")
           
            if currentValue:
                dictNodesProperties[f'{dtc}.{pToC}'] = currentValue
            else:
                dictNodesProperties[f'{dtc}.{pToC}'] = str(0)
                warningText = f'Свойство {dtc}.{pToC} пустое. Заполнено нулями'

            dictNodesProperties[f'{dtc}.{pToC}'] = currentValue
    
        
    #пробежаться по нодам и осуществить замену
    for j, node in enumerate(jsonGraph['graph']['nodes']):
        thisPID = node['primitiveID']
        for property1 in node['properties'].keys():
            if f'{thisPID}.{property1}' in dictNodesProperties.keys():
                for prop in [pt for pt in node['properties'].keys() if pt!=property1]:
                    if prop in dictNodesProperties[f'{thisPID}.{property1}']:
                        spattern = rf"SWT\['.*\.{re.escape(prop)}'\]"
                        if not re.search(spattern, dictNodesProperties[f'{thisPID}.{property1}']):
                            dictNodesProperties[f'{thisPID}.{property1}'] = dictNodesProperties[f'{thisPID}.{property1}'].replace(prop, f"SWT['{thisPID}.{prop}']")


    #################################открываем JSON графа#############3
    calculatedProperties = {k : False for k in dictNodesProperties.keys()}
    global SWT
    SWT = pd.read_json(SWTPath, orient='records', lines = True)
    #planned_at данные
    for col in SWT.columns:
        if col.endswith('planned_at'):
            SWT[col] = pd.to_datetime(SWT[col], errors='coerce').dt.strftime('%d.%m.%Y')
    
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
    #print('!')
    #input(dictNodesProperties)
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
                            #input('here!!!')
                            left_side = graphName
                            megaNode0 = megaNode0.merge(SWT[['_t', cp]], how='left', on = '_t')
                            megaNode0[f"{graphName}.{cp.split('.')[1]}"] = megaNode0[cp]
                            del megaNode0[cp]
                            #megaNode0.rename(columns={cp : f"{left_side}.{cp.split('.')[1]}"})
                            #input('here2!')
                    #ФА
                    if re.search('FactorAnalysis_[0-9]{1,5}.value', cp):
                        nameP = cp.split('.')[0]
                        rightSide = nodes[nodeNumbers[nameP]]['properties']['name']['expression'].replace('"','')
                        megaNode0 = megaNode0.merge(SWT[['_t', cp]], how='left', on = '_t')
                        megaNode0[f"{graphName}.{rightSide}"] = megaNode0[cp]
                        del megaNode0[cp]
                    #ФА
                    #print(megaNode0)
            except KeyError as e:
                #print(e)
                #print(calculatedProperties)
                continue
                
            except TypeError as te:
                #print(errText)
                #input('teloop')
                raise TypeError(f'Ошибка: {str(te)}')
            
            except NameError as ne:
                errText = str(ne)
                #print(f'Порт (выражение) {dictNodesProperties[cp]} не найден в свойстве {cp.split(".")[1]} примитива {cp.split(".")[0]}. Требуется проверить заполнение свойства {cp.split(".")[0]}')
                ne = f'Порт (выражение) {dictNodesProperties[cp]} не найден в свойстве {cp.split(".")[1]} примитива {cp.split(".")[0]} ({nodes[nodeNumbers[cp.split(".")[0]]]["properties"]["name"]["expression"]}). Требуется проверить заполнение свойства {cp}'
                raise NameError(f'Ошибка: {str(ne)}')
                #return False
 
            except Exception as unhandled:
                errText = str(unhandled)
                print(errText)
                print(f'{cp} ; {dictNodesProperties[cp]}')
                raise
                #return False
    #print('Ноды успешно рассчитаны')
    SWTFiltered = SWT[SWT['mY'] == lastPeriod]
    #print('Вставка значений в Data-примитивы...')
    SWTFiltered = SWT[SWT['mY'] == lastPeriod]

    #SWTFiltered.to_json('test.json', orient='records', lines=True)
    if not megaNodeExists:
        megaNode = megaNode0
    else:
        #megaNode.set_index('_t').combine_first(megaNode0.set_index('_t'))
        #megaNode = megaNode.reset_index(drop=True, inplace=True)
        exColumns = [col for col in megaNode0.columns if col!='_t']
        megaNode = megaNode.drop(exColumns, axis=1, errors = 'ignore')
        megaNode = pd.merge(megaNode, megaNode0, how='right', on = ['_t'])
        megaNode = megaNode.reset_index(drop=True)
        
    megaNode.to_json(megaNodePath, orient='records', lines = True, index = False)    
    
    #Убрана часть с Data (была закомеентирована)
    try:
        for j, node in enumerate(jsonGraph['graph']['nodes']):
            for property in node['properties'].keys():
                if f'{node["primitiveID"]}.{property}' in calculatedProperties.keys() or ('DataLakeNode_' in f'{node["properties"][property]["expression"]}' and not 'DataLakeNode_' in node['primitiveID'] or ('megaNode' in node['primitiveID'] and property == 'value')):
                    requiredValue = SWTFiltered[f'{node["primitiveID"]}.{property}'].iloc[0]
                    
                    requiredValue = round(requiredValue,5) if isinstance(requiredValue,float) else requiredValue
                    requiredValue = round(requiredValue,2) if (isinstance(requiredValue,float) and abs(requiredValue) >= 100) else requiredValue
                    

                    #requiredValue = 0 if (requiredValue == -0.0 or requiredValue == 0.0) else requiredValue
                    
                    try:
                        requiredValue = int(requiredValue) if isinstance(requiredValue, np.int64) else requiredValue
                    except AttributeError:
                        requiredValue = int(requiredValue) if not isinstance(requiredValue, float) else requiredValue
                    except UnboundLocalError:
                        if not isinstance(requiredValue, str):
                            requiredValue = int(requiredValue) if not (isinstance(requiredValue, float) or isinstance(requiredValue, int)) else requiredValue
                        else:
                            requiredValue = requiredValue.replace('"','') if requiredValue[0]=='"' else requiredValue
                    #if 'К-т сменяемости' in node['properties']['name']['expression']:
                    #    input(f" {node['properties']['name']['expression']}: {requiredValue}")
                    node['properties'][property]['value'] = requiredValue
                elif isinstance(node['properties'][property]['expression'], str):
                    if not 'if(' in node['properties'][property]['expression']:
                        node['properties'][property]['value'] = node['properties'][property]['expression'].replace('"','')
                        #node['properties'][property]['value'] = node['properties'][property]['expression'][1:len(node['properties'][property]['expression'])-1]
                elif isinstance(node['properties'][property]['expression'], float) or isinstance(node['properties'][property]['expression'], int):
                    node['properties'][property]['value'] = node['properties'][property]['expression']
    except KeyError as ke:
        print(f'Ошибка! {ke} Полный список колонок {SWT.columns}')
        print(f'также словарь {dictNodesProperties}')
        exit(1)

    ###Alert Part###
    if alerted:
        primitiveProperties = []
        for n, node in enumerate(nodes):
            for pty in node['properties'].keys():
                primitiveProperties.append(f"{node['primitiveID']}.{pty}")
                
        alerts = [primitive for primitive in nodeNumbers if 'Alert' in primitive]
        if len(alerts)!=1:
            showerror(f"На примитиве всего {len(alerts)} алертов. Должен быть один.")
            
        alertPrimitive = [primitive for primitive in nodeNumbers if 'Alert' in primitive][0]
        thisAlertPropertiesK, thisAlertProperties = nodes[nodeNumbers[alertPrimitive]]['properties'].keys(),nodes[nodeNumbers[alertPrimitive]]['properties']
        conditionsList = [c for c in thisAlertProperties if re.search('condition_[0-9]{1}', c)]
        conditionsN = {condition : thisAlertProperties[condition]['expression'] for condition in conditionsList}
        #conditionsN = {condition : thisAlertProperties[condition.replace('condition, notification')]['expression'] for condition in conditionsList}
        for k,v in conditionsN.items():
            for pt in primitiveProperties:
                if pt in v:
                    conditionsN[k] = conditionsN[k].replace(pt, f'SWT["{pt}"]')
                            
        for cdk, cdv in conditionsN.items():
            try:
                calculateExpression(SWT, alertPrimitive, cdk, cdv)
                print(f'Условное выражение {alertPrimitive}.{cdk} вычислено')
            except Exception as e:
                showerror(f'Произошла ошибка при вычислении {alertPrimitive}.{cdk} ;{cdv}. Полный текст : {str(e)}"')
        #####часть с root_condition#####
        
        for pc in nodes[nodeNumbers[alertPrimitive]]['properties'].keys():
            rc_expression = nodes[nodeNumbers[alertPrimitive]]['properties']['root_condition']['expression']
            #input(rc_expression)
            if pc in rc_expression and not nodes[nodeNumbers[alertPrimitive]]['properties'][pc]['expression']:
                print(f'{pc} ; 2')
                print(f'Проверьте заполнение свойства {pc} примитива {nodes[nodeNumbers[alertPrimitive]]["primitiveID"]}')
                raise KeyError(f'Проверьте заполнение свойства {pc} примитива {nodes[nodeNumbers[alertPrimitive]]["primitiveID"]}')
        
        try:
            rc_expression = nodes[nodeNumbers[alertPrimitive]]['properties']['root_condition']['expression']
            calculateExpression(SWT, alertPrimitive, 'root_condition', rc_expression)
        except KeyError as kke:
            print(f'В примитиве {alertPrimitive} отсутствует root_condition. Пропускаю...')
            print(rc_expression)
            #input(kke)

        #теперь фильтруем SWT по периоду и вытаскиваем что получилось за соответствующий месяц, пишем в value
        SWTFiltered = SWT[SWT['mY'] == reportMonthPeriod]
        requiredProperties = nodes[nodeNumbers[alertPrimitive]]['properties']
        try:
          for rpk, rpv in requiredProperties.items():
              if 'condition' in rpk:
                  requiredValue = SWTFiltered[f'{alertPrimitive}.{rpk}'].iloc[0]
                  if not isinstance(requiredValue, str):
                      requiredValue = int(requiredValue)
                  jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties'][rpk]['value'] = requiredValue
              else:
                  if isinstance(jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties'][rpk]['expression'], str):
                      #if rpk == 'notification_4': input(jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties'][rpk]['expression'])
                      jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties'][rpk]['value'] = \
                      jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties'][rpk]['expression'].replace('"', '')
                      #if rpk == 'notification_4': input(jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties'][rpk]['value'])
                  else:
                      jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties'][rpk]['value'] = \
                      jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties'][rpk]['expression']
        except KeyError as ke:
            raise(ke)
            print(f'Проверьте заполнение условий в примитиве {alertPrimitive}. Пропускаю значения...')

        #periodMY = SWTFiltered['mY'].iloc[0]
        monNum, yearNum = reportMonthPeriod.split('.')
        monNum = int(monNum)
        periodText = f'{monthes[monNum-1]} {yearNum} г.'
        
        #заплатка для alert_text, он дублирует значения...
        if 'alert_text' in requiredProperties.keys():
             jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties']['alert_text']['value'] = \
             jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties']['root_condition']['value'].replace('"','')
             recommendationText = jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties']['root_condition']['value']
             themeText = jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties']['alert_subject']['value'] + ' за ' + periodText
             emailTo = jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties']['email_address']['value']
        #конец фильтрации
        
        #алертинг альтернативный
        
        currentAlertDF = pd.DataFrame({'alert_subject' : [themeText], 'alert_text' : [recommendationText], 'email_address' : [emailTo]})
        alertJsonPath = os.path.join('/opt', 'otp', 'external_data', 'alerts',f'{graphName}_alert.json')
        #нужно ли посылать
        sendEmail = False
        try:
            lastAlertDF = pd.read_json(alertJsonPath, orient='records', lines = True)
        except FileNotFoundError:
            lastAlertDF = pd.DataFrame(columns = ['col1'])
        if len(lastAlertDF.columns) == 1 or not currentAlertDF.equals(lastAlertDF):
            sendEmail = True
            conditionWorked = True
        
        if sendEmail: 
            currentAlertDF.to_json(alertJsonPath, orient='records', lines = True)
            resSent = sendAlert(themeText, recommendationText, emailTo, graphName)
            print('Алерт успешно отправлен') if resSent else print('Произошла ошибка при отправке алерта')
            #sendemail Сообщение отправлено
            
        print(f'cond Worked : {sendEmail}')
    ####Alert Part End####
    for n, node in enumerate(nodes):
        for pty in node['properties'].keys():
            node['properties'][pty]['status'] = 'complete'
    ####completed####
    
    if len(warningText) >0:
        print(warningText)
    print('Расчет графа успешно завершен')

    print('Ноды успешно рассчитаны')
    SWT.drop(columns=['mY'], inplace=True)
    SWT.to_json(SWTPath, orient='records', lines=True)

    #input(jsonGraph['graph']['nodes'][nodeNumbers[alertPrimitive]]['properties']['notification_4']['value'])
    savedJsonString = json.dumps(jsonGraph)
    bytesJson = savedJsonString.encode()
    #PUT2#
    put1 = session.put(path1, data=bytesJson, headers = headers)
    print(f"Граф-2 {graphToProcess} {'OK' if put1.status_code == 200 else 'error'}")
    put2 = session.put(pathG, data=pathGDict)
    print(f"Суперграф-2 {graphToProcess} {'OK' if put2.status_code == 200 else 'error'}")

    return jsonGraph
