###dataLakeAUTO###
import os,requests,json,pandas as pd
#import OTL
import time, json, re
#from calculateChangeGraphHandler import calculateGraph
import datetime
from datetime import date, datetime as datetimeM

#def unix_to_datetime(unix_time):
#    return datetime.datetime.utcfromtimestamp(unix_time).strftime("%Y-%m")

def showerror(errText):
    print(errText)
    exit(1)

def fillDataLake(graph):
    errText = ''
    megaNodePath = os.path.join('/opt', 'otp', 'external_data', 'megaNode', 'megaNode.json')
    URL,PORT,USERNAME,PASSWORD,session = '127.0.0.1','6081','admin','admin',requests.Session()
    url, body = f'http://{URL}:{PORT}/auth/login', {"login": USERNAME, "password": PASSWORD}
    r = session.post(url, data=body)
    if r.status_code != 200: showerror(f"Ошибка получения токена, код {r.status_code}")
    url = f'http://{URL}:{PORT}/supergraph/v1/fragments'
    r = session.get(url)
    print("Суперграф загружен") if r.status_code == 200 else showerror("Ошибка загрузки суперграфа, код {r.status_code}")
    errors = False
    requiredDictionary,thisJson,pathGDict  = {},{},{}
    path1, found = "", False
    DataLakes = {}
    try:
        megaNodeSWT = pd.read_json(megaNodePath, orient='records', lines = True)
    except:
        print('МегаНода не найдена!')
    #input('тут')
    for i in r.json()['fragments']:
        #print(i)
        if i['name'] == graph:
            id, found = i['id'], True
            path1 = f'http://{URL}:{PORT}/supergraph/v1/fragments/{id}/graph'
            pathG = f'http://{URL}:{PORT}/supergraph/v1/fragments/{id}'
            try:
                thisJsonR = session.get(path1)
                if thisJsonR.status_code != 200:
                    print(f"Код ошибки {thisJsonR.status_code}")
                    exit()
            except:
                print(f"Ошибка открытия сессии")
                exit()
            requiredDictionary['id'],requiredDictionary['name'],pathGDict['name'] = i['id'], i['name'], i['name']
            if not found:
                print("Граф с наименованием {graphName} не найден на сервере")
                return False
            thisJsonJSON = thisJsonR.json()
            for ek in ['status', 'status_code', 'error']:
                del thisJsonJSON[ek]
            for j, node in enumerate(thisJsonJSON['graph']['nodes']):
                if 'DataLake' in node['primitiveID']:
                    DataLakes[i['name']] = node['primitiveID']
                elif 'PeriodSetup' in node['primitiveID']:
                    periodNode = node['primitiveID']
                    startTime = node['properties']['start']['expression'].replace('"','')
                    finishTime = node['properties']['finish']['expression'].replace('"','')

    
    #print(f'{i["name"]}')
    #input(f'loaded : {DataLakes}')
    if not found:
        print(f'Проверьте наличие графа {graph}')
        exit(1)
    #читаем вайд
    wideReadyPath = os.path.join('/opt','otp','external_data', 'wide_ready','wide_ready.json')
    wide = pd.read_json(wideReadyPath, orient='records', lines=True)
    wide.sort_values(by=['_t'], inplace=True)
    wide_tailed = wide.tail(24)
    
    ###time section. для days_for_all_fact и автоматического обнуления###
    #input(0)
    today=date.today()
    #input(1)
    current_year = today.year
    current_month = today.month
    current_day = today.day
    
    if today.day > 20:
        reportMonth = (today.replace(day=1) - datetime.timedelta(days=1)).month
        reportYear = (today.replace(day=1) - datetime.timedelta(days=1)).year
    else:
        reportMonth = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1).month - 1
        reportYear = (today.replace(day=1) - datetime.timedelta(days=1)).year
    
    dt = int(datetimeM(reportYear, reportMonth,1,0,0).strftime('%s'))
    wide_tailed.loc[wide_tailed['_t']>dt,'days_for_all_fact'] = 0
    ###time section end###
    
    for graph in DataLakes.keys():
        #перенесено из прошлого. здесь всегда будет 1 граф, т.е. 1 ключ
        time_begin = time.time()
        
        try:
            SWTFolderPath = os.path.join('/opt','otp','external_data', 'SWT', f"{graph}")
            SWTFilePath = os.path.join('/opt','otp','external_data', 'SWT', f"{graph}", f"{graph}.json")
            
            if not os.path.exists(SWTFolderPath):
                os.makedirs(SWTFolderPath)
                print(f'Создана папка {SWTFolderPath}')
            for file in os.listdir(SWTFolderPath):
                if file.endswith(".json"):
                    os.remove(os.path.join(SWTFolderPath,file))
            # это {DataLakes[graph]} текущая DLnode
            # input({DataLakes[graph]})
            wide_current = wide_tailed.copy(deep=True)
            wide_current = wide_current.reindex(sorted(wide_current.columns), axis=1)

            wide_current = wide_current.add_prefix(f"{DataLakes[graph]}.")
            wide_current = wide_current.rename(columns={f"{DataLakes[graph]}._t" : "_t"})               
            #wide_current['_t'] = wide_current['_t'] + 10800
            #wide_current = wide_current[pd.to_datetime(wide_current['_t'], unit = 's').dt.year == int(startTime[-5:-1])]
            wide_current['_sn'] = 1
            wide_current = wide_current.drop(f"{DataLakes[graph]}._sn", axis=1)
            wide_current = wide_current.tail(24)
            wide_current = wide_current.fillna(0)


            #'budget_value', 'fact_value' было в списке еще
            for j, node in enumerate(thisJsonJSON['graph']['nodes']):
            #убираю external_influence и management_action ; это вычисляемые поля по идее
                llist = ['name', 'value_to_goal', 'switcher', 'colors', 'kir_operation', \
                             'boundary_coef', 'critical_coef',  'switcher', \
                             'efficiency', 'value_to_goal', 'risk_appetite_probability', \
                             'approach', 'business_direction', 'business_process', 'coordinator', 'director', 'risk_owner', \
                             'goal', 'identifier', 'is_compliance', 'is_key_risk', 'is_quantified', 'is_sanc_sens', 'is_typical', 'risk_type',\
                             'residual_risk_probability', 'current_risk_probability', 'consequences', 'related_risk', 'management_strategy', 'management_resources', \
                             'cost', 'comment', 'planned_at', 'status', 'description', 'risk_fact_value']
                        #wide_current[f"{node['primitiveID']}.{rProperty}"] = node['properties'][rProperty]['expression']
                #for pty in node['properties'].keys():
                #    if 'notification' in pty or 'risk_appetite' in pty:
                #        llist.append(pty)                        

                #if 'Risk_' in node['primitiveID']:
                #    for rProperty in ['approach', 'business_direction', 'business_process', 'coordinator', 'director', \
                #    'goal', 'identifier', 'is_compliance', 'is_key_risk', 'is_quantified', 'is_sanc_sens', 'is_typical', 'risk_type']:
                #        wide_current[f"{node['primitiveID']}.{rProperty}"] = node['properties'][rProperty]['expression']
                             
                if 'Data_' in node['primitiveID']:
                    thisExpression = node['properties']['value']['expression'].split('.')[1] if '.' in node['properties']['value']['expression'] \
                                     else (node['properties']['value']['expression'] if node['properties']['value']['expression'] else 0)
                                     
                    if 'DataLakeNode_' in node['properties']['value']['expression']:
                        if f"{DataLakes[graph]}.{node['properties']['value']['expression'].split('.')[1]}" not in wide_current.columns:
                            errText = f"Проверьте наличие показателя {node['properties']['value']['expression'].split('.')[1]} в вайде. Нода {node['primitiveID']} с именем {node['properties']['name']['expression']}"
                            print([col for col in wide_current.columns if 'days' in col])
                            raise TypeError(errText)
                            
                    for column in wide_current.columns:
                        if '.' in column:
                            if thisExpression == column.split('.')[1]:
                                wide_current[f"{node['primitiveID']}.value"] = wide_current[column]
                                break
                            elif node['properties']['value']['expression']:
                                wide_current[f"{node['primitiveID']}.value"] = node['properties']['value']['expression']
                            else:
                                wide_current[f"{node['primitiveID']}.value"] = ''
                if 'megaNode' in node['primitiveID']:
                    if not node['properties']['value']['expression'] or not node['properties']['name']['expression']:
                        print(f'В меганоде {node["primitiveID"]} не заполнены выражения имени и значения! Необходимо исправить')
                        exit(1)
                    else:
                        try:  
                            requiredMegaNodeColumn = node['properties']['value']['expression']
                            megaNodeSWT0 = megaNodeSWT[['_t',requiredMegaNodeColumn]]
                            wide_current = wide_current.merge(megaNodeSWT0, how = 'left', on = '_t')
                            wide_current[f"{node['primitiveID']}.value"] = wide_current[requiredMegaNodeColumn]
                            wide_current = wide_current.drop(f"{requiredMegaNodeColumn}", axis=1)
                        except:
                            showerror(f'Проверьте заполнение МегаНоды {node["primitiveID"]}\nФормат: ИмяГрафа.ИмяСвойства')

                if 'Measures_' in node['primitiveID']:
                    msListNum=0
                    for np, port in enumerate(node['initPorts']):
                        if 'inPort' in port['primitiveName']:
                            msListNum+=1
                    if msListNum == 0:
                        for mV in ['efficiency', 'value']:
                            
                            #print(f' {mV} : {type(mV)}')
                            #input(node['properties'][mV]['expression'])
                            try:
                                if isinstance(node['properties'][mV]['expression'], str):
                                    node['properties'][mV]['expression'] = float(node['properties'][mV]['expression'].replace(' ',''))
                                    #print(type(node['properties'][mV]['expression']))
                            except:
                                llist.remove(mV)
                            
                            if not isinstance(node['properties'][mV]['expression'], str):
                                wide_current[f"{node['primitiveID']}.value"] = node['properties'][mV]['expression']
                            #if isinstance(node['properties'][mV]['expression'],str):
                            #    print(llist)
                            #    if 'efficiency' in llist: llist.remove('efficiency')
                            #    #llist.append('value')
                            #elif (isinstance(node['properties'][mV]['expression'],float) or isSinstance(node['properties'][mV]['expression'],int)):
                            #    if not 'efficiency' in llist: llist.append('efficiency')
                        #added 270424
                    if msListNum > 0 and re.search('\+|-|\*|/',node['properties']['efficiency']['expression']):
                        try:
                            llist.remove('efficiency')
                            llist.remove('value')
                        except:
                            pass

                #input(llist)
                for ppt in llist:
                    if ppt in node['properties'].keys() and (not isinstance(node['properties'][ppt]['expression'], int) and \
                        not isinstance(node['properties'][ppt]['expression'], float)):
                            if not 'if(' in node['properties'][ppt]['expression']:
                                wide_current[f"{node['primitiveID']}.{ppt}"] = node['properties'][ppt]['expression'].replace('"', '').replace("'","")
                    elif ppt in node['properties'].keys() and (isinstance(node['properties'][ppt]['expression'], int)  or \
                        isinstance(node['properties'][ppt]['expression'], float)):
                        wide_current[f"{node['primitiveID']}.{ppt}"] = node['properties'][ppt]['expression']
                
                #Риск аппеетит- специально
                if re.search(node['primitiveID'], 'RiskAppetite_[0-9]{1,5}'):
                    wide_current[f"{node['primitiveID']}.{description}"] = node['properties'][description]['expression'].replace('"', '')
                               
            wide_current = wide_current[[col for col in wide_current.columns if not col.startswith('DataLakeNode_')]]             
            #когда пустой- это самый плохой вариант

            wide_current[f'{periodNode}.start'] = f'{startTime}'
            # if startTime[0] != "0" else f'0{startTime}'
            wide_current[f'{periodNode}.finish'] = f'{finishTime}'
            # if startTime[0] != "0" else f'0{startTime}'
            
            wide_current.to_json(SWTFilePath, orient='records', lines=True)

            time_end = time.time()
            seconds = time_end - time_begin
            print(f"{graph} : нода {DataLakes[graph]} успешно обновлена, время обновления {seconds} c.")
            return True
            
        except Exception as e:
            print(f"Ошибка обновления ноды {DataLakes[graph]} графа {graph}, завершение работы")
            print(e)
            if errText: 
                raise TypeError(errText)
            else:
                return False

