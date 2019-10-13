import os
import numpy as np
from matplotlib import pyplot as plt

'''
filecount: count of csv file = count of evaluated pictures
'''
def generate_graph_data_and_median_mean(filecount:int, max_queries:int, ref_name:str):
    graphs = np.zeros((filecount, max_queries))
    for idx, f in enumerate(os.listdir("evaluation_results/current_csv/{}".format(ref_name))):
        points = np.loadtxt(os.path.join("evaluation_results/current_csv/{}".format(ref_name), f), delimiter=',')
        xp = points[:,1]
        fp = points[:,2]
        x = np.arange(1, max_queries+1)
        graph = np.interp(x, xp, fp)
        graphs[idx] = graph

    #calc median and mean values of best and last iteration
    file = open("evaluation_results/{}_mean_median.txt".format(ref_name), "w")
    file.write('median of last iteration: {} \n'.format(np.median(graphs[:, -1])))
    file.write('mean of last iteration: {} \n'.format(np.mean(graphs[:, -1])))
    file.write('median of best iteration: {} \n'.format(np.median(np.min(graphs, axis=1))))
    file.write('mean of best iteration: {} \n'.format(np.mean(np.min(graphs, axis=1))))
    file.close()

    mean_graph = graphs.mean(axis=0)
    plt.plot(mean_graph)
    plt.ylabel("mean of thresholds")
    plt.xlabel("queries")
    plt.savefig("evaluation_results/{}_graph".format(ref_name))
    plt.close()
    return mean_graph

if __name__ == '__main__':

    graph_names = ["mnist_bapp_100000_MSE_defgan_orig_noDefense_100","mnist_bapp_100000_MSE_defgan_100"]
    mean_graphs = []
    for graph_name in graph_names:
        mean_graphs.append(generate_graph_data_and_median_mean(100, 100000, graph_name))

    fig, ax = plt.subplots()
    i = 0
    for g in mean_graphs:
        if i == 0:
            ax.plot(g, 'tab:blue', linewidth=1.0)
        elif i == 1:
            ax.plot(g, 'tab:orange', linewidth=1.0) #green
        elif i == 2:
            ax.plot(g, 'tab:orange', linewidth=1.0)
        i+=1

    ax.set_yscale('log')
    import matplotlib.ticker as ticker


    def format(x, pos):
        return '%1.6f' % (x)

    formatter = ticker.FuncFormatter(format)
    ax.yaxis.set_major_formatter(formatter)
    #ax.yaxis.set_minor_formatter(formatter)
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=[1,5]))
    #ax.yaxis.set_minor_locator(ticker.AutoLocator())
    #ax.yaxis.set_major_locator(ticker.FixedLocator([0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]))

    plt.grid(which='both')
    #adapt this when generating new graphs
    plt.title("Defense-GAN on MNIST")
    plt.legend(["Without defense", "With defense"])
    plt.ylabel("mean of reached perturbations")
    plt.xlabel("number of queries")
    plt.tight_layout()

    plt.savefig("evaluation_results/{}_comparison_graph".format(graph_names[0]))
    print("saved to evaluation_results/{}_comparison_graph".format(graph_names[0]))
