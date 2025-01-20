from datasets import Dataset
from rich import print


def load_data(file_path, test=False):
    records = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("IN:") and "OUT:" in line:
                input = line.split("IN:")[1].split("OUT:")[0].strip()
                output = line.split("OUT:")[1].strip()
                if test:
                    records.append(
                        {
                            "conversations": [
                                {
                                    "role": "system",
                                    "content": "You translate commands into actions",
                                },
                                {"role": "user", "content": f"<cmd>{input}</cmd>"},
                            ],
                            "output": output,
                        }
                    )
                else:
                    records.append(
                        {
                            "conversations": [
                                {
                                    "role": "system",
                                    "content": "You translate commands into actions",
                                },
                                {"role": "user", "content": f"<cmd>{input}</cmd>"},
                                {
                                    "role": "assistant",
                                    "content": f"<action>{output}</action>",
                                },
                            ]
                        }
                    )
    return Dataset.from_list(records)


if __name__ == "__main__":
    dataset = load_data("data/simple_split/tasks_train_simple.txt")
    print(dataset)
    print(dataset[0])
    print(dataset[0]["conversations"])
    print(dataset[0]["conversations"][1])
