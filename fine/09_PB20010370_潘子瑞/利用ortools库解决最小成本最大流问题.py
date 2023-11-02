import numpy as np

from ortools.graph.python import min_cost_flow
from ortools.graph.python import max_flow

def main():
    smcf = min_cost_flow.SimpleMinCostFlow()
    smf = max_flow.SimpleMaxFlow()

    start_nodes= np.loadtxt('test data/start_nodes.txt',comments='#',encoding='utf-8')#读取起点
    end_nodes = np.loadtxt('test data/end_nodes.txt',comments='#',encoding='utf-8')#读取终点
    capacities = np.loadtxt('test data/capacities.txt',comments='#',encoding='utf-8')#起点和终点之间的路径容量
    all_arcs = smf.add_arcs_with_capacity(start_nodes, end_nodes, capacities)
    status = smf.solve(0, 4)
    m=smf.optimal_flow()
    unit_costs = np.loadtxt('test data/unit_costs.txt',comments='#',encoding='utf-8')#单位费用
    # Define an array of supplies at each node.
    supplies = [int(m), 0, 0, 0, -int(m)]

    # Add arcs, capacities and costs in bulk using numpy.
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, capacities, unit_costs)

    # Add supply for each nodes.
    smcf.set_nodes_supply(np.arange(0, len(supplies)), supplies)

    # Find the min cost flow.
    status = smcf.solve()

    if status != smcf.OPTIMAL:
        print('There was an issue with the min cost flow input.')
        print(f'Status: {status}')
        exit(1)
    print('Max flow:', m)
    print(f'Minimum cost: {smcf.optimal_cost()}')
    print('')
    print(' Arc    Flow / Capacity Cost')
    solution_flows = smcf.flows(all_arcs)
    costs = solution_flows * unit_costs
    for arc, flow, cost in zip(all_arcs, solution_flows, costs):
        print(
            f'{smcf.tail(arc):1} -> {smcf.head(arc)}  {flow:3}  / {smcf.capacity(arc):3}       {cost}'
        )


if __name__ == '__main__':
    main()