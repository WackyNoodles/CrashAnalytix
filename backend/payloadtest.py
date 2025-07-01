def payload(accident_output, entities_result, severity_result, regnum_list):
    entity_mapping = {
        0: 'pedestrian',   # person
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    severity_mapping = {
        0: 'Critical',
        1: 'Minor',
        2: 'Moderate'
    }

    accident_result, file_path = accident_output

    if accident_result != 0:
        return {"result": "No Accident Detected"}

    result = {
        "result": "Accident Detected",
        "snapshot_path": file_path,
        "severity": severity_mapping.get(severity_result, "Unknown"),
        "entities": []
    }

    reg_index = 0

    for code in entities_result:
        entity_type = entity_mapping.get(code, "unknown")
        entity_obj = {"type": entity_type}

        if entity_type in ['car', 'motorcycle', 'bus', 'truck']:
            if reg_index < len(regnum_list):
                entity_obj["license_plate"] = regnum_list[reg_index]
                reg_index += 1
            else:
                entity_obj["license_plate"] = ""

        result["entities"].append(entity_obj)

    return result
# accident_output = [0, "filepath"]
# entities_result = [7, 2, 0]  # truck, car, pedestrian
# severity_result = 1  # Minor
# regnum_list = ["NWFP-893", ""]

# print(payload(accident_output, entities_result, severity_result, regnum_list))
