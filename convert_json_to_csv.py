import os
import json
import csv
from collections import defaultdict

def convert_json_to_csv(output_directory, log_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 找出 log 目录下所有 JSON 文件
    json_files = [f for f in os.listdir(log_directory) if f.endswith('.json')]

    for json_file_name in json_files:
        json_file_path = os.path.join(log_directory, json_file_name)
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        
        trace_events = data.get('traceEvents', [])
        
        output_sub_dir = os.path.join(output_directory, os.path.splitext(json_file_name)[0])
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
        
        categorized_events = defaultdict(list)
        
        for event in trace_events:
            cat = event.get('cat')
            if cat:
                # 展开args字段
                if 'args' in event:
                    for key, value in event['args'].items():
                        event[key] = value
                    del event['args']
                
                categorized_events[cat].append(event)
        
        for cat, events in categorized_events.items(): 
            csv_file_name = os.path.join(output_sub_dir, f"{cat}.csv")
            
            header = set()
            for event in events:
                header.update(event.keys())
            
            header = list(header)

            with open(csv_file_name, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)
                for event in events:
                    row = [json.dumps(event.get(h, '')) if isinstance(event.get(h, ''), (dict, list)) else event.get(h, '') for h in header]
                    csv_writer.writerow(row)
        
        print(f"文件 {json_file_name} 已成功转化为 {output_sub_dir} 目录下的csv")

    print("所有文件转换完成。")
