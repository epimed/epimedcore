from pymongo import MongoClient
from neo4j import GraphDatabase, basic_auth

class Mongodb:
    
    def __init__(self, host: str = 'localhost', port: int = 27017):
        self.host = host
        self.port = port
        self.client = MongoClient(host, port)
        
    def find_one(self, database, collection, filters): 
        return self.client[database][collection].find_one(filters)
    
    def find(self, database, collection, filters): 
        return self.client[database][collection].find(filters)
    
    def count(self, database, collection, filters): 
        return self.client[database][collection].count(filters)
    
    def distinct (self, database, collection, filters, field):
        return self.client[database][collection].find(filters).distinct(field)
    

class Neo4j:
    
    def __init__(self):
        self.driver = None
        self.session = None
    
    def open_session(self):
        self.driver = GraphDatabase.driver("bolt://epimed-db.u-ga.fr:7687", auth=basic_auth("epimed", "Ymlututpd9"))
        self.session = self.driver.session()
    
    def close_session(self):
        self.session.close()
        self.driver.close()
        
    def execute_query(self, query):
        self.open_session()
        result = self.session.run(query)
        nodes = []
        for record in result:
            node = record['n']
            nodes.append(node)
        self.close_session()
        return nodes
        
    def get_gene_symbol(self, id_gene):
        # Query for gene symbol
        query ="MATCH (n:Gene) WHERE n.uid=" + str(id_gene) + " RETURN n"
        nodes = self.execute_query(query)
        if nodes:
            gene_symbol = nodes[0]['gene_symbol']
            return gene_symbol
        else:
            return str(id_gene)

    def get_gene_node(self, id_gene):
        # Query for gene symbol
        query ="MATCH (n:Gene) WHERE n.uid=" + str(id_gene) + " RETURN n"
        # print(query)
        nodes = self.execute_query(query)
        if nodes:
            return nodes[0]
        else:
            return 0