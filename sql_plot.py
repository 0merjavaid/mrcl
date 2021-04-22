import logging
import os

logger = logging.getLogger('experiment')

import itertools
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import experiment as exp
from utils import utils
import configs.plots.sql as reg_parser
import ast
import mysql.connector
import time
import random

results_dict = {}
std_dict = {}
all_experiments = []
folders = []
all_experiments = {}
legend = []
colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k'))



def main():

    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    run = p.parse_known_args()[0].run
    all_args = vars(p.parse_known_args()[0])

    args = utils.get_run(all_args, run)

    my_experiment = exp.experiment(args["name"], args, args["output_dir"], sql=False,
                               run=int(run / total_seeds),
                               seed=total_seeds)

    my_experiment.results["all_args"] = all_args

    with open("credentials.json") as f:
        db_data = json.load(f)

    db_name = args["database"]
    while (True):
        try:
            conn = mysql.connector.connect(
                host=db_data['database'][0]["ip"],
                user=db_data['database'][0]["username"],
                password=db_data['database'][0]["password"]
            )
            break
        except:
            time.sleep((random.random() + 0.2) * 5)

    sql_run = conn.cursor()
    sql_run.execute("USE " + args["database"] + ";")
    print(args['query'])
    qry = args['query']
    labels = qry[qry.find("SELECT") + 6: qry.find("FROM")].split(",")
    print(labels)

    output = sql_run.execute(args['query'])
    output = sql_run.fetchall()
    results_dict = {}
    print(output)
    for row in output:
        # print(row)
        if len(row) > 2:
            if row[2:len(row)] in results_dict:
                results_dict[row[2:len(row)]][0].append(row[0])
                results_dict[row[2:len(row)]][1].append(row[1])
            else:
                results_dict[row[2:len(row)]] = [[row[0]], [row[1]]]
        else:
            if "single" in results_dict:
                results_dict["single"][0].append(row[0])
                results_dict["single"][1].append(row[1])

            else:
                results_dict["single"] = [[row[0]], [row[1]]]

    a = int(np.ceil(np.sqrt(len(results_dict))))
    fig, axs = plt.subplots(a, a, sharex=True, sharey=False)

    inner_counter, outer_counter = 0, 0
    for cur_exp in results_dict:
        if a == 1:
            axs.plot(results_dict[cur_exp][0], results_dict[cur_exp][1])
        else:
            axs[inner_counter, outer_counter].plot([x for x in results_dict[cur_exp][0]],results_dict[cur_exp][1] )
            axs[inner_counter, outer_counter].set_title(cur_exp)

        inner_counter +=1
        # outer_counter += 1
        if inner_counter == a:
            inner_counter=0
            outer_counter += 1
    if len(labels) > 2:
        fig.suptitle(labels[2:len(labels)])
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.tight_layout()
    plt.show()



    print(my_experiment.path + "result.pdf")
    plt.savefig(my_experiment.path + "result.pdf", format="pdf")

if __name__ == "__main__":
    main()