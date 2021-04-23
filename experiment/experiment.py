import argparse
import datetime
import json
import logging
import os
import subprocess
from logging import handlers
import mysql.connector
import sys
import time
import random
import json




logger = logging.getLogger('experiment')



# How to use this class (See the main function at the end of this file for an actual example)

## 1. Create an experiment object in the main file (the one used to run the experiment)
## 2. Pass a name and args_parse objecte. Output_dir corresponds to directly where all the results will be stored
## 3. Use experiment.path to get path to the output_dir to store any other results (such as saving the model)
## 4. You can also store results in experiment.result dictionary (Only add objects which are json serializable)
## 5. Call experiment.store_json() to store/update the json file (I just call it periodically in the training loop)

class experiment:
    '''
    Class to create directory and other meta information to store experiment results.
    A directory is created in output_dir/DDMMYYYY/name_0
    In-case there already exists a folder called name, name_1 would be created.

    Race condition:
    '''

    def __init__(self, name, args, output_dir="../", sql=True, run=None, seed=None):
        import sys
        self.sql = sql
        if sql:
            with open("credentials.json") as f:
                self.db_data = json.load(f)
            #
            self.db_name = "umer" + name
            while(True):
                try:
                    conn = mysql.connector.connect(
                        host=self.db_data['database'][0]["ip"],
                        user=self.db_data['database'][0]["username"],
                        password=self.db_data['database'][0]["password"]
                    )
                    break
                except:
                    time.sleep((random.random() + 0.2) * 5)

            sql_run = conn.cursor()
            try:
                sql_run.execute("CREATE DATABASE "+self.db_name +";")
            except:
                logger.info("DB already exists")
            sql_run.execute("USE "+ self.db_name + ";")

            conn.close()


        if name[-1] != "/":
            name += "/"

        self.command_args = "python " + " ".join(sys.argv)
        self.run = run
        self.name_initial = name

        if not args is None:
            if run is not None:
                self.name = name + str(run) + "/" + str(seed)
            else:
                self.name = name
            self.params = args
            print(self.params)
            self.results = {}
            self.dir = output_dir

            root_folder = datetime.datetime.now().strftime("%d%B%Y")

            if not os.path.exists(output_dir + root_folder):
                try:
                    os.makedirs(output_dir + root_folder)
                except:
                    assert (os.path.exists(output_dir + root_folder))

            self.root_folder = output_dir + root_folder
            full_path = self.root_folder + "/" + self.name

            ver = 0

            while True:
                ver += 1
                if not os.path.exists(full_path + "_" + str(ver)):
                    try:
                        os.makedirs(full_path +  "_" + str(ver))
                        break
                    except:
                        pass
            self.path = full_path + "_" + str(ver) + "/"
            args["output_dir"] = self.path
            if sql:
                self.database_path =  os.path.join(self.root_folder, self.name_initial, "results.db")


                if sql:

                    ret = self.make_table("runs", args, ["run"])
                    self.insert_value("runs", args)
                    if ret:
                        print("Table created")
                    else:
                        print("Table already exists")

            fh = logging.FileHandler(self.path + "log.txt")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter('run:' + str(args['run']) + ' ' + name + ' %(levelname)-8s %(message)s'))
            logger.addHandler(fh)

            ch = logging.handlers.logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(
                logging.Formatter('run:' + str(args['run']) + ' ' + name + ' %(levelname)-8s %(message)s'))
            logger.addHandler(ch)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False


            self.store_json()

    def get_connection(self):
        if self.sql:
            while (True):
                try:
                    conn = mysql.connector.connect(
                        host=self.db_data['database'][0]["ip"],
                        user=self.db_data['database'][0]["username"],
                        password=self.db_data['database'][0]["password"]
                    )
                    break
                except:
                    time.sleep((random.random() + 0.2) * 5)

            sql_run  = conn.cursor()
            sql_run.execute("USE " + self.db_name + ";")
            return conn, sql_run
        return None

    def is_jsonable(self, x):
        try:
            json.dumps(x)
            return True
        except:
            return False


    def make_table(self, table_name, data_dict, primary_key):
        if self.sql:
            conn, sql_run = self.get_connection()

            table = "CREATE TABLE " + table_name + " ("
            counter = 0
            for a in data_dict:
                if type(data_dict[a]) is int or type(data_dict[a]) is float:
                    table = table + a + " real"
                else:
                    table = table + a + " text"

                counter += 1
                if counter != len(data_dict):
                    table += ", "
            if primary_key is not None:
                table += " ".join([",", "PRIMARY KEY(", ",".join(primary_key)]) + ")"
            table = table + ");"
            print(table)
            try:
                sql_run.execute(table)
                conn.commit()
                conn.close()
                return True
            except:
                conn.close()
                logger.error("Not making table");
                return False
        return None


    def insert_value(self, table_name, data_dict):
        if self.sql:
            conn, sql_run = self.get_connection()
            query = " ".join(["INSERT INTO", table_name,   str(tuple(data_dict.keys())).replace("'", ""),   "VALUES", str(tuple(data_dict.values()))]) + ";"
            # print(query)
            sql_run.execute(query)
            conn.commit()
            conn.close()


    def insert_values(self, table_name, keys, value_list):
        if self.sql:
            conn, sql_run = self.get_connection()
            strin = "("
            counter = 0
            for a in value_list[0]:
                counter+=1
                strin += "%s"
                if counter != len(value_list[0]):
                    strin +=","
            strin += ");"

            query = " ".join(
                ["INSERT INTO", table_name, str(tuple(keys)).replace("'", ""), "VALUES", strin])

            # print(value_list)
            print(query, value_list)
            # quit()
            sql_run.executemany(query, value_list)
            conn.commit()
            conn.close()


    def add_result(self, key, value):
        assert (self.is_jsonable(key))
        assert (self.is_jsonable(value))
        self.results[key] = value

    def store_json(self):
        pass
        # self.conn.commit()
        # with open(self.path + "metadata.json", 'w') as outfile:
        #     json.dump(self.__dict__, outfile, indent=4, separators=(',', ': '), sort_keys=True)
        #     outfile.write("")

    def get_json(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=True)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='iCarl2.0')
#     parser.add_argument('--batch-size', type=int, default=50, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--epochs', type=int, default=200, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--epochs2', type=int, default=10, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--lrs', type=float, nargs='+', default=[0.00001],
#                         help='learning rate (default: 2.0)')
#     parser.add_argument('--decays', type=float, nargs='+', default=[0.99, 0.97, 0.95],
#                         help='learning rate (default: 2.0)')
#     # Tsdsd
#
#     args = parser.parse_args()
#     e = experiment("TestExperiment", args, "../../")
#     e.add_result("Test Key", "Test Result")
#     e.store_json()
