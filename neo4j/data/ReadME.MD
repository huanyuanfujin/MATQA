[数据库构建]
    neo4j-admin import --mode=csv --nodes=entity.csv --relationships=roles.csv --database=graph.db
[启动数据库服务]
    neo4j.bat console

[entity.csv]  实体
[roles.csv] 关系


