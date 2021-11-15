from mlp_data_gen import load_mlp_instance_from_fpath

from mlp_cplex import create_model_A, parse_solution_model_A
from docplex.mp.progress import ProgressDataRecorder
import pandas as pd
from docplex.util.status import JobSolveStatus

TIME_LIMIT=60*60*2
    
def solve_from_fpath(fpath):
    nodes, cost_matrix = load_mlp_instance_from_fpath(fpath)
    model = create_model_A(nodes, cost_matrix)
    model.parameters.timelimit=TIME_LIMIT
    # accept approximations.
    model.parameters.mip.tolerances.mipgap=0
    prog_recorder = ProgressDataRecorder()
    model.add_progress_listener(prog_recorder)
    solution = model.solve(log_output=True)
    
    return solution, prog_recorder

def progress_to_dataframe(progress_recorder):
    colnames = ['id', 'has_incumbent',
                'current_objective', 'best_bound', 'mip_gap',
                'current_nb_iterations', 'current_nb_nodes', 'remaining_nb_nodes',
                'time', 'det_time']
    df = pd.DataFrame(list(tuple(d) for d in progress_recorder.recorded), columns=colnames)
    
    return df

def evaluate(fpath):
    solution, progress = solve_from_fpath(fpath)
    if solution.solve_status in [JobSolveStatus.OPTIMAL_SOLUTION, JobSolveStatus.FEASIBLE_SOLUTION]:
        permutation = parse_solution_model_A(solution)
        value = solution.objective_value
        is_optimal = solution.solve_status == JobSolveStatus.OPTIMAL_SOLUTION
        with open(fpath + '.solution.txt', 'w') as f:
            f.write('\n'.join(permutation))
            f.write(f'\n{value}')
            f.write(f'\n{is_optimal}')

    df = progress_to_dataframe(progress)
    df.to_csv(fpath + '.progress.csv')

if __name__ == '__main__':
    import sys, os
    fpath = sys.argv[1]
    assert os.path.exists(fpath)
    evaluate(fpath)