import re
from py2neo import Graph
import base64
graph = Graph('http://localhost:7474/db/data/')


def query(name, req=''):
    # name = '基于Neo4J知识图谱的查询命令'
    if len(req) == 0:
        data = graph.run("match(p:s {name:'%s'}) -[r]->(n) return p.name, r, n.name limit 50" % ''.join(name))
    else:
        data = graph.run("match(p:s {name:'%s'}) -[r:%s]->(n) return p.name, r, n.name limit 50" % (name, req))

    data = list(data)
    return get_json_data(data)


def get_json_data(data):
    """
    查询结果解析，熟悉Neo4J的应该很清楚，这是基本功，自己查询的时候，多打印一些中间结果，看一看就
    知道查询出来的是什么玩意，程序里面为什么要那样解析。
    :param data:
    :return:
    """
    json_data = {'data': [], "links": []}
    d = []

    for i in data:
        d.append(i['p.name']+"_")
        d.append(i['n.name']+"_")
        d = list(set(d))
    name_dict = {}
    count = 0
    for j in d:
        j_array = j.split("_")

        data_item = {}
        name_dict[j_array[0]] = count
        count += 1
        data_item['name'] = j_array[0]
        json_data['data'].append(data_item)
    for i in data:
        string = str(i['r'])
        result = string.split('-[:')[1].split(' {}]->(')[0]
        # print('result:', string)
        link_item = {}
        try:
            link_item['target'] = name_dict[i['n.name']]
            link_item['source'] = name_dict[i['p.name']]
            link_item['value'] = result
            json_data['links'].append(link_item)
        except Exception as ex:
            pass
    return json_data


