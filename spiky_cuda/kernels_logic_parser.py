import re
import sys
import os


def remove_comments(text):
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove single-line comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    return text


def find_matching_brace(text, start):
    brace_count = 1
    pos = start
    while brace_count > 0 and pos < len(text):
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1
    return pos if brace_count == 0 else -1


def parse_function(func_text):
    # Extract function name and arguments
    match = re.match(r'(\w+)\s*\((.*?)\)\s*{', func_text, re.DOTALL)
    if match:
        func_name, args = match.groups()
        # Find the matching closing brace
        end_pos = find_matching_brace(func_text, match.end())
        if end_pos > 0:
            # Keep the original body text without stripping whitespace
            body = func_text[match.end():end_pos-1]
            return func_name, args, body
    return None


def format_args(args, add_reference_symbols):
    # Split args on commas, strip whitespace, and rejoin with proper spacing
    formatted = []
    for arg in args.split(','):
        parts = arg.strip().split()
        # Handle pointer types
        name_part = parts[-1].strip()
        if add_reference_symbols:
            name_part = '&' + name_part
        if '*' in arg:
            type_part = ' '.join(parts[:-1]).strip()
            formatted.append(f"{type_part} {name_part}")
        else:
            formatted.append(f"{' '.join(parts[:-1])} {name_part}")
    return ',\n    '.join(formatted)


def get_param_names(args):
    # Extract just the parameter names from the argument list
    param_names = []
    for arg in args.split(','):
        parts = arg.strip().split()
        if len(parts) >= 2:
            # Get the last part which is the parameter name
            name = parts[-1].strip()
            # Remove any * from the name
            name = name.replace('*', '')
            name = name.replace('&', '')
            param_names.append(name)
    return param_names


def generate_cu_from_proto(input_file, output_file):
    print(f"Processing {input_file} -> {output_file}")
    
    with open(input_file, 'r') as f:
        content = f.read()
        print(f"Read {len(content)} bytes from {input_file}")
        # Remove comments before processing
        content = remove_comments(content)

    # Find all function definitions
    functions = []
    pos = 0
    while pos < len(content):
        match = re.search(r'\b\w+\s*\(.*?\)\s*{', content[pos:], re.DOTALL)
        if not match:
            break
        
        start = pos + match.start()
        end = find_matching_brace(content, pos + match.end())
        if end < 0:
            break
            
        functions.append(content[start:end])
        pos = end
    
    print(f"Found {len(functions)} functions")
    
    try:
        with open(output_file, 'w') as f:
            f.write("#undef ATOMIC\n")
            for func in functions:
                parsed = parse_function(func)
                if parsed:
                    func_name, args, body = parsed
                    new_func_name = f"PFX({func_name})"
                    new_on_cpu_wrapper_func_name = f"PFX({func_name + '_on_cpu_wrapper'})"
                    formatted_args = format_args(args, True)
                    formatted_on_cpu_wrapper_args = format_args(args, False)
                    if len(formatted_args.strip()) == 0:
                        raise RuntimeError(f'function {func_name} without args')
                    cuda_args = ",\n    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx" if func_name.endswith('_logic') else ''
                    # Keep the body exactly as it was in source
                    kernel_logic_prefix = 'KERNEL_LOGIC_ONLY_HOST_PREFIX' if 'ATOMIC' in body else 'KERNEL_LOGIC_PREFIX'
                    f.write(f"{kernel_logic_prefix} void {new_func_name}(\n    {formatted_args}{cuda_args}\n)\n{{{body.replace('ATOMIC_PFX','PFX')}}}\n\n")
                    if func_name.endswith('_logic'):
                        param_names = get_param_names(args)
                        param_names_str = ', '.join(param_names)
                        f.write(f"{kernel_logic_prefix} void {new_on_cpu_wrapper_func_name}(\n    {formatted_on_cpu_wrapper_args}{cuda_args}\n)\n")
                        f.write("{\n")
                        f.write(f"    PFX({func_name})({param_names_str}, blockIdx, blockDim, threadIdx);\n")
                        f.write("}\n\n")
                    print(f"Processed function {func_name}")

            f.write("#ifndef NO_CUDA\n")
            f.write("#define ATOMIC\n")
            for func in functions:
                parsed = parse_function(func)
                if parsed:
                    func_name, args, body = parsed
                    if 'ATOMIC' in body:
                        new_func_name = f"PFX({func_name}_atomic_)"
                        formatted_args = format_args(args, True)
                        if len(formatted_args.strip()) == 0:
                            raise RuntimeError(f'function {func_name} without args')
                        cuda_args = ",\n    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx" if func_name.endswith('_logic') else ''
                        # Keep the body exactly as it was in source
                        f.write(f"KERNEL_LOGIC_ATOMIC_PREFIX void {new_func_name}(\n    {formatted_args}{cuda_args}\n)\n{{{body}}}\n\n")
                        print(f"Generated atomic variant of function {func_name}")

            f.write("#undef ATOMIC\n")
            # Generate CUDA wrapper functions
            for func in functions:
                parsed = parse_function(func)
                if parsed:
                    func_name, args, body = parsed
                    atomic_postfix = '_atomic_' if 'ATOMIC' in body else ''
                    if func_name.endswith('_logic'):
                        param_names = get_param_names(args)
                        param_names_str = ', '.join(param_names)
                        f.write(f"__global__ void PFX({func_name}_cuda)(\n    {format_args(args, False)}\n)\n")
                        f.write("{\n")
                        f.write(f"    PFX({func_name}{atomic_postfix})({param_names_str}, blockIdx, blockDim, threadIdx);\n")
                        f.write("}\n\n")
            
            f.write("#endif\n")
        print(f"Successfully wrote to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {str(e)}")


def main(input_file, output_file):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return
    
    generate_cu_from_proto(input_file, output_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python kernels_logic_parser.py <input.proto> <output.cu>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
