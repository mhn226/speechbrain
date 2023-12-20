import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scp_file",
        required=True,
        type=str,
        help="Path to the scp file for which you want to replace the ark path.",
    )
    parser.add_argument(
        "--new_ark_path",
        required=True,
        type=str,
        help="The path you want to use as replacement towards the ark file.",
    )
    args = parser.parse_args()

    # Reading the lines
    lines = []
    with open(args.scp_file, "r") as scp_file:
        for line in scp_file:
            # filtering last line
            if line != "\n":
                line_split = line.split(" ")
                identifier = line_split[0]
                bytes = line_split[1].split(":")[1]

                lines.append(f"{identifier} {args.new_ark_path}:{bytes}")
            else:
                lines.append(line)

    with open(args.scp_file, "w") as scp_file:
        for line in lines:
            scp_file.write(line)
