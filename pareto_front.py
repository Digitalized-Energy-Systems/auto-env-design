import numpy as np
import pandas as pd
from scipy.spatial import distance


def is_dominated(p1, p2):
    """
    Check if point p1 is dominated by point p2.
    """
    return all(p2 <= p1) and any(p2 < p1)


def compute_pareto_front(df):
    """
    Identify non-dominated points (Pareto front).
    """
    pareto_mask = np.ones(df.shape[0], dtype=bool)
    for i, row in df.iterrows():
        for j, other_row in df.iterrows():
            if is_dominated(row, other_row):
                pareto_mask[i] = False
                break
    return pareto_mask


def compute_fuzzy_dominance(df, pareto_front, epsilon):
    """
    Compute fuzzy dominance based on distance to Pareto front.
    A point is fuzzy dominated if:
    1. It is strictly dominated (dominated is True).
    2. The distance to the Pareto front is less than or equal to epsilon.
    """
    fuzzy_dominated_mask = np.zeros(df.shape[0], dtype=bool)
    pareto_points = df[pareto_front].values  # Non-dominated points (Pareto front)
    
    for i, row in df.iterrows():
        if not ~pareto_front[i]:  # Only check for dominated points
            continue
        
        # Compute Euclidean distance between the current point and all Pareto front points
        distances = distance.cdist([row.values], pareto_points, metric='euclidean')
        
        # If the minimum distance to the Pareto front is <= epsilon, mark as fuzzy dominated
        if np.min(distances) <= epsilon:
            fuzzy_dominated_mask[i] = True

    return fuzzy_dominated_mask


def main():
    # Sample data
    data = {
        'metric1': [1, 2, 3, 4, 5, 3, 2, 1, 4, 5, 5],
        'metric2': [5, 4, 3, 2, 1, 2, 3, 4, 1, 2, 5]
    }

    df = pd.DataFrame(data)

    # Step 1: Compute Pareto dominance
    pareto_mask = compute_pareto_front(df)

    # Step 2: Add a 'dominated' column (inverse of Pareto mask)
    df['dominated'] = ~pareto_mask

    # Step 3: Compute fuzzy dominance with epsilon
    epsilon = 1  # Example value for epsilon
    fuzzy_dominated_mask = compute_fuzzy_dominance(df[['metric1', 'metric2']], pareto_mask, epsilon)

    # Step 4: Add 'fuzzy_dominated' column
    df['fuzzy_dominated'] = fuzzy_dominated_mask

    # Display result
    print(df)


if __name__ == '__main__':
    main()
