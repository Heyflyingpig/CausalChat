"""
处理llm的输出，提取json对象
string -> json
"""
import json
import logging
def excute_output(llm_output: str) -> dict:
    try:
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise json.JSONDecodeError("No JSON object found in LLM output", llm_output, 0)
        json_part = llm_output[json_start:json_end]
        parsed_json = json.loads(json_part)
        return parsed_json
    except json.JSONDecodeError as e:
        logging.error(f"Could not parse JSON from LLM response: {e}\nRaw response: {llm_output}")
        return {}