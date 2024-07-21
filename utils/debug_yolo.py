import pandas as pd
from datetime import timedelta
import numpy as np

def debug_results_yolo(csv_path=''):
    # Load the data into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Add the new 'direction' column based on 'distance_to_center'
    df['trayectory'] = df['distance_to_center'].apply(lambda x: 'positive' if x >= 0 else 'negative')

    # Add the 'in_out_status' column based on 'distance_to_center'
    df['in_out_status'] = (df['distance_to_center'] >= 0).astype(int)

    # Add the 'in_out_status_label' column with labels 'Inside' or 'Outside'
    df['in_out_status_label'] = df['in_out_status'].replace({1: 'Inside', 0: 'Outside'})

    # Function to convert frame_number to hh:mm:ss format rounded down to the nearest second
    def frame_to_time(frame_number):
        # Assuming 15 frames per second
        total_seconds = frame_number // 15
        # Use divmod to get hours, minutes, and seconds
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        # Format as hh:mm:ss
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Add the 'time' column to df
    df['time'] = df['frame_number'].apply(frame_to_time)

    # Function to detect specific transition patterns for Possible ID Switch
    def detect_specific_transition(group):
        first_sign = np.sign(group.iloc[0]['distance_to_center'])
        last_sign = np.sign(group.iloc[-1]['distance_to_center'])

        if first_sign != last_sign:
            return False

        group['status_change'] = group['in_out_status_label'].ne(group['in_out_status_label'].shift()).cumsum()
        segments_summary = group.groupby('status_change')['in_out_status_label'].agg(['first', 'count'])
        segments_list = list(segments_summary.itertuples(index=False, name=None))

        desired_patterns = [
            [('Inside', 10), ('Outside', 10), ('Inside', 10)],
            [('Outside', 10), ('Inside', 10), ('Outside', 10)]
        ]

        def pattern_exists(pattern):
            pattern_idx = 0
            for segment in segments_list:
                if pattern_idx >= len(pattern):
                    break
                if segment[0] == pattern[pattern_idx][0] and segment[1] >= pattern[pattern_idx][1]:
                    pattern_idx += 1
            return pattern_idx == len(pattern)

        return any(pattern_exists(pattern) for pattern in desired_patterns)

    # Function to calculate the number of transitions for each ID
    def calculate_transitions(group):
        group = group.sort_values('frame_number')
        signs = np.sign(group['distance_to_center'])
        sign_changes = (signs.diff() != 0) & signs.diff().notnull()
        return sign_changes.sum()

    # Function to calculate the occlusion index for each ID
    def calculate_occlusion_index(group):
        nearest_frames = group.iloc[(group['distance_to_center'].abs()).argsort()[:10]]
        occlusion_index = nearest_frames['overlap'].mean() if not nearest_frames.empty else 0
        return occlusion_index

    # Calculate aggregated information including transitions, potential ID switches, and First Frame Number
    def calculate_aggregated_info(group):
        number_of_frames = group['frame_number'].nunique()
        first_frame_number = group['frame_number'].min()
        first_frame_time = group['time'].min()
        event_duration = number_of_frames // 15
        transition_count = calculate_transitions(group)
        occlusion_index = calculate_occlusion_index(group)
        is_switch_risk = detect_specific_transition(group)

        first_status = group.nsmallest(1, 'frame_number')['in_out_status_label'].iloc[0]
        last_status = group.nlargest(1, 'frame_number')['in_out_status_label'].iloc[0]
        trajectory = 'not defined'
        if first_status == 'Inside' and last_status == 'Outside':
            trajectory = 'leaving'
        elif first_status == 'Outside' and last_status == 'Inside':
            trajectory = 'entering'

        if is_switch_risk:
            category = 'Possible ID Switch'
        elif trajectory == 'not defined':
            category = 'Not Defined'
        elif (trajectory == 'entering' or trajectory == 'leaving') and occlusion_index < 0.3:
            category = 'Perfect In' if trajectory == 'entering' else 'Perfect Out'
        else:
            category = 'Needs Review'

        # Get the most frequent direction for the ID
        direction = group['direction'].mode()[0] if not group['direction'].mode().empty else np.nan

        return pd.Series({
            'Number of Frames': number_of_frames,
            'Event Duration': event_duration,
            'First Frame Number': first_frame_number,
            'First Frame Time': first_frame_time,
            'Trajectory': trajectory,
            'Transition Count': transition_count,
            'Occlusion Index': occlusion_index,
            'Category': category,
            'Direction': direction  # Include the direction column
        })

    # Reset index to ensure 'id' is not both an index level and a column label
    df = df.reset_index(drop=True)

    # Apply the function to each ID group and create the aggregated DataFrame
    aggregated_info = df.groupby('id').apply(calculate_aggregated_info).reset_index()

    # Count the occurrences of each category
    category_counts = aggregated_info['Category'].value_counts()

    # Calculate the percentage of each category
    category_percentages = (category_counts / aggregated_info.shape[0]) * 100

    # Combine counts and percentages into a summary DataFrame
    category_summary = pd.DataFrame({
        'Total Values': category_counts,
        'Percentage': category_percentages
    }).reset_index().rename(columns={'index': 'Category'})

    unique_id_counts = df.groupby('direction')['id'].nunique()

    return category_summary, unique_id_counts


if __name__ == '__main__':
    # Example usage
    csv_path = r'C:\Users\joaqu\Desktop\MIVO\Demo\apumanque_entrada_2_20240707_0900_short_condensed_bbox_yolo7.csv'
    debug_results_yolo(csv_path)
