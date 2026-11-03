"""MedAgentSim integration excerpts. See THIRD_PARTY_NOTICES.md."""
import time
from medsim.query_model import BAgent
class MeasurementAgent:
    def __init__(self, backend_str="gpt4") -> None:
        # language model backend for measurement agent
        self.backend = backend_str
        self.pipe = BAgent(model_name=backend_str)

    def update_scenario(self, scenario):
        self.agent_hist = ""
        # presentation information for measurement 
        self.presentation = ""
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
    def inference_measurement(self, question) -> str:
        answer = str()
        answer = self.pipe.query_model("\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()

def compare_results(diagnosis, correct_diagnosis, mod_pipe, similarity_threshold=0.8, tries=3, timeout=5.0):
    """
    Compares the doctor's diagnosis with the correct diagnosis using a similarity-based approach.

    Args:
        diagnosis (str): The diagnosis provided by the doctor.
        correct_diagnosis (str): The correct diagnosis for the case.
        mod_pipe (BAgent): The initialized moderator instance.
        similarity_threshold (float): Threshold for similarity to decide "Yes". Defaults to 0.8.
        tries (int): Number of retry attempts. Defaults to 3.
        timeout (float): Time in seconds between retries. Defaults to 5.0.

    Returns:
        tuple: (decision (str), similarity (float))
    """
    prompt = (
        f"Here is the correct diagnosis: {correct_diagnosis}\n"
        f"Here was the doctor's diagnosis: {diagnosis}\n"
        f"Rate the similarity between the two diagnoses on a scale of 0 to 1, where 0 means completely dissimilar and 1 means identical. "
        f"Based on the similarity score, decide whether they match. Respond strictly in the format:\n"
        f"[0.XX]\n"
        f"Do not include any additional text or explanation."
    )
    system_prompt = (
        "You are a medical moderator responsible for assessing similarity between two diagnoses. "
        "Respond strictly in the format [0.XX] where 0.XX is similarity. Do not include any extra text."
    )
    # breakpoint()
    for attempt in range(tries):
        try:
            # Query the model
            response = mod_pipe.query_model(prompt=prompt, system_prompt=system_prompt, tries=1, timeout=timeout)
            print(f"Attempt {attempt + 1} response: {response}")

            # Extract response
            if response.startswith("[") and response.endswith("]"):
                similarity_str = response[1:-1]  # Remove square brackets
                # similarity_str, decision_str = response_content.split(",")
                
                # Parse similarity and decision
                similarity = float(similarity_str)
                # decision = decision_str.split(":")[1].strip().lower()

                # Validate response
                if 0 <= similarity <= 1:
                    return similarity>=similarity_threshold
                else:
                    print(f"Invalid similarity score: {similarity}")

            else:
                print(f"Response not in expected format: {response}")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")

        # Wait before retrying
        time.sleep(timeout)

    print("Failed to compare results after multiple attempts; marking diagnosis incorrect.")
    return False
