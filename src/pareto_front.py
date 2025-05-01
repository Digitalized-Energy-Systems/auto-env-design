import numpy as np
import pandas as pd


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


def point_to_segment_distance(point, segment_start, segment_end):
    """
    Compute the shortest distance from a point to a line segment (segment_start, segment_end).
    """
    # Convert the inputs to numpy arrays
    point = np.array(point)
    segment_start = np.array(segment_start)
    segment_end = np.array(segment_end)

    # Vector from segment_start to segment_end
    segment_vector = segment_end - segment_start

    # Vector from segment_start to the point
    point_vector = point - segment_start

    # Project point_vector onto the segment_vector
    segment_length_sq = np.dot(segment_vector, segment_vector)

    # Handle the case where the segment_start and segment_end are the same point
    if segment_length_sq == 0.0:
        return np.linalg.norm(point_vector)

    t = np.dot(point_vector, segment_vector) / segment_length_sq
    t = max(0, min(1, t))  # Ensure that t is between 0 and 1, meaning the projection is within the segment

    # Compute the closest point on the segment
    projection = segment_start + t * segment_vector

    # Compute the distance between the point and the projection on the segment
    return np.linalg.norm(projection - point)


def compute_fuzzy_dominance(df, pareto_front, epsilon):
    if isinstance(epsilon, (int, float)):
        return compute_fuzzy_dominance_single(df, pareto_front, epsilon)
    elif isinstance(epsilon, (list, tuple, np.ndarray)):
        return compute_fuzzy_dominance_multiple(df, pareto_front, epsilon)


def compute_fuzzy_dominance_single(df, pareto_front, epsilon):
    """
    Compute fuzzy dominance based on the distance to the Pareto front line.
    A point is fuzzy dominated if:
    1. It is strictly dominated.
    2. The shortest distance to the line connecting the Pareto front points is less than or equal to epsilon.
    """
    fuzzy_dominated_mask = np.zeros(df.shape[0], dtype=bool)

    # Extract non-dominated points (Pareto front) and sort them
    pareto_points = df[pareto_front].values
    pareto_points = pareto_points[pareto_points[:, 0].argsort()]

    for i, row in df.iterrows():
        if not ~pareto_front[i]:  # Only check for dominated points
            continue

        # Check the distance from the point to all line segments on the Pareto front
        min_distance = np.inf
        for j in range(len(pareto_points) - 1):
            segment_start = pareto_points[j]
            segment_end = pareto_points[j + 1]

            # Calculate distance from the point to the current segment
            distance_to_segment = point_to_segment_distance(row.values, segment_start, segment_end)
            min_distance = min(min_distance, distance_to_segment)

        # Special case: Only one point on pareto front
        if len(pareto_points) == 1:
            segment_start = pareto_points[0]
            segment_end = pareto_points[0]
            min_distance = point_to_segment_distance(row.values, segment_start, segment_end)

        # If the minimum distance to the Pareto front is <= epsilon, mark as fuzzy dominated
        if min_distance <= epsilon:
            fuzzy_dominated_mask[i] = True

    return fuzzy_dominated_mask


def compute_fuzzy_dominance_multiple(df, pareto_front, epsilons):
    fuzzy_dominated_mask = np.zeros(df.shape[0], dtype=bool)
    pareto_points = df[pareto_front].values  # Non-dominated points (Pareto front)

    for i, row in df.iterrows():
        if not ~pareto_front[i]:  # Only check for dominated points
            continue

        point = np.array(row.values)
        fuzzy_point_best = point - np.array(epsilons)

        # Check if fuzzy point is still dominated by any Pareto point
        # If not, the point is fuzzy dominated
        fuzzy_dominated_mask[i] = ~np.any(np.array([is_dominated(fuzzy_point_best, pareto_point) for pareto_point in pareto_points]))

    return fuzzy_dominated_mask


# def main():
#     # Sample data
#     data = {
#         'metric1': [1, 2, 3, 4, 5, 3, 2, 1, 4, 5, 5],
#         'metric2': [5, 4, 3, 2, 1, 2, 3, 4, 1, 2, 5]
#     }

#     df = pd.DataFrame(data)

#     # Step 1: Compute Pareto dominance
#     pareto_mask = compute_pareto_front(df)

#     # Step 2: Add a 'dominated' column (inverse of Pareto mask)
#     df['dominated'] = ~pareto_mask

#     # Step 3: Compute fuzzy dominance with epsilon
#     epsilon = 1  # Example value for epsilon
#     fuzzy_dominated_mask = compute_fuzzy_dominance(df[['metric1', 'metric2']], pareto_mask, epsilon)

#     # Step 4: Add 'fuzzy_dominated' column
#     df['fuzzy_dominated'] = fuzzy_dominated_mask

#     # Display result
#     print(df)


# if __name__ == '__main__':
#     main()
