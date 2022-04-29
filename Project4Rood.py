##Adam S. Rood
##CSE 545-50 (Artificial Intelligence)
##Project #4
##09/21/20

##import packages
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation
import math
import datetime
import statistics as stats
import pickle
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) 

##load tsp datafile
n_tsp = 100

tsp = [tuple(x) for x in round(pd.read_csv('Random100.tsp', 
        skiprows = 7, header = None, 
        delimiter = ' ')[[1,2]],6).to_records(index = False)]

##static graph for report visualizations
def create_graph_static(list_name):
    plt.scatter(np.array([x[0] for x in tsp]), np.array([x[1] for x in tsp]), s = 20, c = 'olive')
    plt.plot([x[0] for x in list_name], [x[1] for x in list_name], c = 'g')
    plt.fill([x[0] for x in list_name], [x[1] for x in list_name], c = 'lemonchiffon')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.xlim(min([x[0] for x in tsp]) - 10, max([x[0] for x in tsp]) + 10)
    plt.ylim(min([x[1] for x in tsp]) - 10, max([x[1] for x in tsp]) + 10)
    plt.suptitle('Traveling Salesman Problem', fontsize = 16)
    plt.title('Genetic Algorithm (n = ' + str(n_tsp) + ')', fontsize = 10)
    plt.show()

##create n x n cost matrix
def create_cost_matrix(tsp):
    cost_matrix = []
    for x in range(len(tsp)):
        for y in range(len(tsp)):
            cost_matrix.append(distance.euclidean(tsp[x],tsp[y]))
    return np.array(cost_matrix).reshape((len(tsp),len(tsp)))

##create cost function 
def cost(cost_mat, route):
    return cost_mat[np.roll(route, 1), route].sum()

##create initial population
def create_population(n):
    global pop
    pop = []
    for x in range(n):
        pop.append([list(np.random.choice(n_tsp, n_tsp, replace = False))])
    df = pd.DataFrame(pop,columns=['key'])
    df['score'] = df.apply(lambda x: cost(cost_matrix, x['key']), axis = 1)
    df['prob'] = pd.Series((1/df['score']))/sum((1/pd.Series(df['score'])))
    return df

##crossover method #1
def crossover_order_one(df, threshold, cost_matrix, mutation_rate):
    parents = np.random.choice(df['key'], size = 2, p = df['prob'], replace = False)
    slicers = list(np.random.choice(n_tsp,2, replace = False))
    slicers.sort()
    thresh = np.random.uniform(0,1,1)
    children = []
    if thresh <= threshold:
        child = parents[0][slicers[0]:slicers[1] + 1]
        leftovers = [x for x in parents[1] if x not in child]
        for x in range(int(slicers[0])):
            child.insert(x, leftovers.pop(0))
        child = child + leftovers
        mutation(child, mutation_rate)
        if cost(cost_matrix,child) < cost(cost_matrix,parents[0]):
            children.append(child)
        else:
            children.append(parents[0])
        child = parents[1][slicers[0]:slicers[1] + 1]
        leftovers = [x for x in parents[0] if x not in child]
        for x in range(int(slicers[0])):
            child.insert(x, leftovers.pop(0))
        child = child + leftovers
        mutation(child, mutation_rate)
        if cost(cost_matrix,child) < cost(cost_matrix,parents[1]):
            children.append(child)
        else:
            children.append(parents[1])
    return children

##crossover method #2
def crossover_order_one_reverse(df, threshold, cost_matrix, mutation_rate):
    parents = np.random.choice(df['key'], size = 2, p = df['prob'], replace = False)
    slicers = list(np.random.choice(n_tsp,2, replace = False))
    slicers.sort()
    thresh = np.random.uniform(0,1,1)
    children = []
    if thresh <= threshold:
        child = parents[0][slicers[0]:slicers[1] + 1]
        child.reverse()
        leftovers = [x for x in parents[1] if x not in child]
        leftovers.reverse()
        for x in range(int(slicers[0])):
            child.insert(x, leftovers.pop(0))
        child = child + leftovers
        mutation(child, mutation_rate)
        if cost(cost_matrix,child) < cost(cost_matrix,parents[0]):
            children.append(child)
        else:
            children.append(parents[0])
        child = parents[1][slicers[0]:slicers[1] + 1]
        child.reverse()
        leftovers = [x for x in parents[0] if x not in child]
        leftovers.reverse()
        for x in range(int(slicers[0])):
            child.insert(x, leftovers.pop(0))
        child = child + leftovers
        mutation(child, mutation_rate)
        if cost(cost_matrix,child) < cost(cost_matrix,parents[1]):
            children.append(child)
        else:
            children.append(parents[1])
    return children

##mutation function
def mutation(child, rate):
    if np.random.choice([0,1], size = 1, p = [1 - rate, rate]) == 1:
        slicers = list(np.random.choice(n_tsp - 1,2, replace = False))
        child[slicers[0]], child[slicers[0] + 1] = child[slicers[0] + 1], child[slicers[0]]
    return child

##breeding function
def breed(df, pop_size, crossover_method, cost_matrix, threshold, mutation_rate):
    new_population = []
    while len(new_population) < pop_size:
        new_population.append(crossover_method(df,threshold,cost_matrix,mutation_rate))
        new_population = [[item] for sublist in new_population for item in sublist]
    df = pd.DataFrame(new_population, columns=['key'])
    df['score'] = df.apply(lambda x: cost(cost_matrix, x['key']), axis = 1)
    df['prob'] = pd.Series((1/df['score']))/sum((1/pd.Series(df['score'])))
    return df, pop[pop['score'] == min(pop['score'])].values.tolist()

##animate the GUI
def animate(num):
    global nn
    if nn < len(for_graph) - 1:
        ax.clear()
        if nn > 0:
            G.remove_edges_from(for_graph[nn - 1])
        G.add_edges_from(for_graph[nn])
        cost_pit = for_graph[nn]
        c = cost(cost_matrix,cost_pit)
        plt.suptitle('Traveling Salesman Problem (TSP) -- Genetic Algorithm', fontsize = 22)
        plt.xlabel('X-Coordinate')
        plt.ylabel('Y-Coordinate')
        ax.set_title('Total distance traveled: ' + str(round(c,2)) + ' miles', fontsize = 16)
        labels = {}
        for x in range(len(tsp)):
            labels[x] = x
        nx.draw_networkx(G, pos, ax = ax, node_color = 'palegreen', with_labels = True)
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        nn += 1

##this is where the program starts
start_time = datetime.datetime.now()
cost_matrix = create_cost_matrix(tsp)
run_log = []
master_log = []
for y in range(50):
    print ('Iteration #',y + 1)
    pop = create_population(200)
    iter_start_time = datetime.datetime.now()
    for x in range(8000):
        pop, minimum = breed(pop, 200, crossover_order_one, cost_matrix, 0.90, 0.05)
        run_log.append(minimum)
        if x % 1000 == 0:
            print('Generations:',x)
    master_log.append(minimum[0])
    iter_run_time = datetime.datetime.now() - iter_start_time
    print(iter_run_time)
run_time = datetime.datetime.now() - start_time

##create a list to analyze results of route lengths
# find_stats = []
# for x in range(len(master_log)):
#     find_stats.append(master_log[x][1])

##summary stats
# print('mean:',np.mean(find_stats))
# print('median:',np.median(find_stats))
# print('mode:',stats.mode(find_stats))
# print('min:',min(find_stats))
# print('max:',max(find_stats))
# print('std:',np.std(find_stats))

##create cost vs. generation graph
# plt.plot(np.arange(0,len(example_coo_100),1), [x[0][1] for x in example_coo_100], label = 'COO (100)')
# plt.plot(np.arange(0,len(example_coo_200),1), [x[0][1] for x in example_coo_200], label = 'COO (200)')
# plt.plot(np.arange(0,len(example_coor_100),1), [x[0][1] for x in example_coor_100], label = 'COOR (100)')
# plt.plot(np.arange(0,len(example_coor_200),1), [x[0][1] for x in example_coor_200], label = 'COOR (200)')
# plt.xlabel('Generation')
# plt.ylabel('Cost')
# plt.legend()
# plt.xlim(0, 8000)
# plt.ylim(0,5000)
# plt.suptitle('Cost vs. Generation', fontsize = 16)

# with open('pickled_data.pkl', 'wb') as file:
#     pickle.dump(master_master_edge_list, file)

# with open('pickled_data_all.pkl', 'wb') as file:
#     pickle.dump(run_log, file)


##load pickled files to save time
master_master_edge_list = pickle.load(open('pickled_data.pkl', 'rb'))
all_data = pickle.load(open('pickled_data_all.pkl', 'rb'))

##create data for animated graph (reduce set size down to only animate
## improved routes)
a_data = []
temp_max = 99999
for x in range(len(all_data)):
    if all_data[x][0][1] < temp_max:
        temp_max = all_data[x][0][1]
        a_data.append(all_data[x][0])

a_data_map = [x[0] for x in a_data]

for_graph = []
for y in range(len(a_data_map)):
    temp_list = []
    for x in range(len(a_data_map[y]) - 1):
        temp_list.append((a_data_map[y][x],a_data_map[y][x+1]))
    temp_list.append((a_data_map[y][x+1], a_data_map[y][0]))
    for_graph.append(temp_list)

##create base graph for GUI
G = nx.Graph()
for i, x in enumerate(tsp):
    G.add_node(i, pos=(x[0],x[1]))
pos = nx.get_node_attributes(G, 'pos')
fig, ax = plt.subplots(figsize=(10, 10))
nn = 0

##launch GUI animation
ani = matplotlib.animation.FuncAnimation(fig, animate, frames = 1000, save_count = 300, interval=100)
plt.show()

##output animated gif 
writergif = matplotlib.animation.PillowWriter(fps=5) 
ani.save('tsp_100_example.gif', writer=writergif)

##create more static graphs for report
# #optimal solution
# list_name = []
# for x in a_data[284][0]:
#     list_name.append(tsp[x])
# list_name.append(tsp[a_data[284][0][0]])
# create_graph_static(list_name)

# #graphs
# list_name = []
# for x in minimum[0][0]:
#     list_name.append(tsp[x])
# list_name.append(tsp[minimum[0][0][0]])
# create_graph_static(list_name)