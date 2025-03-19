import re
import struct
import argparse
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union




# Instruction format types
class Format(Enum):
    BRANCH = auto()    # For branch instructions (call, b, beq, bgt, ret, nop)
    REGISTER = auto()  # For register-register operations
    IMMEDIATE = auto() # For register-immediate operations including memory ops

# Instruction definition class
@dataclass
class InstructionDef:
    mnemonic: str
    opcode: int
    format: Format
    operands: int  # Number of operands

# Token types
class TokenType(Enum):
    LABEL = auto()
    INSTRUCTION = auto()
    REGISTER = auto()
    IMMEDIATE = auto()
    MEMORY_REF = auto()
    DELIMITER = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line_num: int

@dataclass
class Instruction:
    mnemonic: str
    operands: List[Token]
    address: int
    line_num: int
    label: Optional[str] = None

class AssemblerError(Exception):
    def __init__(self, message: str, line_num: int = None):
        self.message = message
        self.line_num = line_num
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        if self.line_num:
            return f"Line {self.line_num}: {self.message}"
        return self.message

class SimpleRiscAssembler:
    # Constants for validation
    MAX_REGISTER = 15
    MIN_SIGNED_IMM = -32768
    MAX_SIGNED_IMM = 32767
    MAX_UNSIGNED_IMM = 65535
    MAX_MEMORY_OFFSET = 32767
    MIN_MEMORY_OFFSET = -32768
    
    def __init__(self):
        # Initialize instruction definitions
        self.instructions = self._initialize_instructions()
        self.symbol_table = {}
        self.current_address = 0
        self.binary_output = []
    
    def validate_register(self, reg_num: int, line_num: int) -> None:
        """Validate register number is within valid range."""
        if not (0 <= reg_num <= self.MAX_REGISTER):
            raise AssemblerError(f"Invalid register number {reg_num}. Must be between 0 and {self.MAX_REGISTER}", line_num)
    
    def validate_immediate(self, imm: int, signed: bool, line_num: int) -> None:
        """Validate immediate value based on whether it's signed or unsigned."""
        if signed:
            if not (self.MIN_SIGNED_IMM <= imm <= self.MAX_SIGNED_IMM):
                raise AssemblerError(
                    f"Signed immediate value {imm} out of range. Must be between {self.MIN_SIGNED_IMM} and {self.MAX_SIGNED_IMM}",
                    line_num
                )
        else:
            if not (0 <= imm <= self.MAX_UNSIGNED_IMM):
                raise AssemblerError(
                    f"Unsigned immediate value {imm} out of range. Must be between 0 and {self.MAX_UNSIGNED_IMM}",
                    line_num
                )
    
    def validate_memory_offset(self, offset: int, line_num: int) -> None:
        """Validate memory offset is within valid range."""
        if not (self.MIN_MEMORY_OFFSET <= offset <= self.MAX_MEMORY_OFFSET):
            raise AssemblerError(
                f"Memory offset {offset} out of range. Must be between {self.MIN_MEMORY_OFFSET} and {self.MAX_MEMORY_OFFSET}",
                line_num
            )
        
    def _initialize_instructions(self) -> Dict[str, InstructionDef]:
        """Initialize the instruction set definitions as per documentation."""
        instructions = {}
        # Register format instructions (3-address ALU instructions)
        instructions["add"] = InstructionDef("add", 0b00000, Format.REGISTER, 3)
        instructions["sub"] = InstructionDef("sub", 0b00001, Format.REGISTER, 3)
        instructions["mul"] = InstructionDef("mul", 0b00010, Format.REGISTER, 3)
        instructions["div"] = InstructionDef("div", 0b00011, Format.REGISTER, 3)
        instructions["mod"] = InstructionDef("mod", 0b00100, Format.REGISTER, 3)
        instructions["and"] = InstructionDef("and", 0b00110, Format.REGISTER, 3)
        instructions["or"]  = InstructionDef("or",  0b00111, Format.REGISTER, 3)
        instructions["lsl"] = InstructionDef("lsl", 0b01010, Format.REGISTER, 3)
        instructions["lsr"] = InstructionDef("lsr", 0b01011, Format.REGISTER, 3)
        instructions["asr"] = InstructionDef("asr", 0b01100, Format.REGISTER, 3)
        
        # 2-address instructions (also use register/immediate format)
        instructions["cmp"] = InstructionDef("cmp", 0b00101, Format.REGISTER, 2)
        instructions["not"] = InstructionDef("not", 0b01000, Format.REGISTER, 2)
        instructions["mov"] = InstructionDef("mov", 0b01001, Format.IMMEDIATE, 2)
        
        # Branch format instructions (0/1-address)
        instructions["nop"] = InstructionDef("nop", 0b01101, Format.BRANCH, 0)
        instructions["ret"] = InstructionDef("ret", 0b10100, Format.BRANCH, 0)
        instructions["beq"] = InstructionDef("beq", 0b10000, Format.BRANCH, 1)
        instructions["bgt"] = InstructionDef("bgt", 0b10001, Format.BRANCH, 1)
        instructions["b"]   = InstructionDef("b",   0b10010, Format.BRANCH, 1)
        instructions["call"]= InstructionDef("call",0b10011, Format.BRANCH, 1)
        
        # Memory operations (use immediate format)
        instructions["ld"]  = InstructionDef("ld",  0b01110, Format.IMMEDIATE, 2)
        instructions["st"]  = InstructionDef("st",  0b01111, Format.IMMEDIATE, 2)
        
        return instructions
    
    def _tokenize_line(self, line: str, line_num: int) -> List[Token]:
        """Convert a line of assembly into tokens."""
        # Remove comments
        line = re.sub(r'/\*.*?\*/', '', line)  # Remove block comments
        
        # Handle @ style comments
        if '@' in line:
            line = line[:line.index('@')]
        
        # Handle # style comments
        if '#' in line:
            line = line[:line.index('#')]
            
        tokens = []
        # Check for label
        label_match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*):(.*)$', line)
        if label_match:
            label, line = label_match.groups()
            tokens.append(Token(TokenType.LABEL, label, line_num))
        
        # Strip leading/trailing whitespace
        line = line.strip()
        if not line:
            return tokens
        
        # Tokenize the instruction and operands
        # Split by commas and whitespace but preserve memory references like imm[rs1]
        instruction_parts = []
        current_part = ""
        in_brackets = False
        for char in line:
            if char == '[':
                in_brackets = True
                current_part += char
            elif char == ']':
                in_brackets = False
                current_part += char
            elif (char == ',' or char.isspace()) and not in_brackets:
                if current_part:
                    instruction_parts.append(current_part)
                    current_part = ""
            else:
                current_part += char
        
        if current_part:
            instruction_parts.append(current_part)
            
        # Remove any empty parts
        instruction_parts = [part.strip() for part in instruction_parts if part.strip()]
        
        if not instruction_parts:
            return tokens
            
        # First part is the instruction
        tokens.append(Token(TokenType.INSTRUCTION, instruction_parts[0].lower(), line_num))
        
        # Process operands
        for part in instruction_parts[1:]:
            if not part:
                continue
                
            # Register?
            reg_match = re.match(r'^r(\d+)$|^(sp|ra)$', part, re.IGNORECASE)
            if reg_match:
                reg_num = 0
                if reg_match.group(1):
                    reg_num = int(reg_match.group(1))
                elif reg_match.group(2) == 'sp':
                    reg_num = 14  # sp = r14
                elif reg_match.group(2) == 'ra':
                    reg_num = 15  # ra = r15
                tokens.append(Token(TokenType.REGISTER, str(reg_num), line_num))
                continue
            
            # Memory reference? Format: imm[rs1]
            mem_match = re.match(r'^([-+]?\d+|0x[0-9a-fA-F]+)?\s*\[\s*r(\d+|sp|ra)\s*\]$', part, re.IGNORECASE)
            if mem_match:
                offset = mem_match.group(1) or '0'
                reg = mem_match.group(2)
                
                # Convert register name to number
                if reg == 'sp':
                    reg = '14'
                elif reg == 'ra':
                    reg = '15'
                
                # Add offset as immediate token
                if offset.startswith('0x'):
                    offset_value = int(offset, 16)
                else:
                    offset_value = int(offset)
                tokens.append(Token(TokenType.IMMEDIATE, str(offset_value), line_num))
                
                # Add register token
                tokens.append(Token(TokenType.REGISTER, reg, line_num))
                continue
            
            # Immediate value?
            imm_match = re.match(r'^([-+]?\d+)$|^(0x[0-9a-fA-F]+)$', part)
            if imm_match:
                value = part
                if part.startswith('0x'):
                    # Keep as hex string for special handling later
                    value = part
                tokens.append(Token(TokenType.IMMEDIATE, value, line_num))
                continue
            
            # Must be a label reference
            tokens.append(Token(TokenType.IMMEDIATE, part, line_num))
        
        return tokens
    
    def _parse_file(self, filename: str) -> List[Instruction]:
        """Parse the assembly file into a list of instructions."""
        instructions = []
        
        try:
            with open(filename, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    tokens = self._tokenize_line(line, line_num)
                    if not tokens:
                        continue
                    
                    # Process label if present
                    label = None
                    if tokens[0].type == TokenType.LABEL:
                        label = tokens[0].value
                        self.symbol_table[label] = self.current_address
                        tokens = tokens[1:]
                    
                    # Skip if only a label
                    if not tokens:
                        continue
                    
                    # Verify instruction
                    if tokens[0].type != TokenType.INSTRUCTION:
                        raise SyntaxError(f"Line {line_num}: Expected instruction, got {tokens[0].value}")
                    
                    mnemonic = tokens[0].value
                    mnemonic = mnemonic.strip().lower()
                    if mnemonic not in self.instructions:
                        raise SyntaxError(f"Line {line_num}: Unknown instruction '{mnemonic}'")
                    
                    # Create instruction object
                    instruction = Instruction(
                        mnemonic=mnemonic,
                        operands=tokens[1:],
                        address=self.current_address,
                        line_num=line_num,
                        label=label
                    )
                    
                    # Validate operand count
                    inst_def = self.instructions[mnemonic]
                    expected_operands = inst_def.operands
                    actual_operands = len(tokens) - 1  # Subtract instruction token
                    
                    if actual_operands != expected_operands and expected_operands != 0:
                        # Special case for memory operations which might have combined operands
                        if not (inst_def.mnemonic in ["ld", "st"] and actual_operands > expected_operands):
                            print(f"Warning: Line {line_num}: Expected {expected_operands} operands for '{mnemonic}', got {actual_operands}")
                    
                    instructions.append(instruction)
                    self.current_address += 4  # Each instruction is 4 bytes
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return []
        except Exception as e:
            print(f"Error parsing file: {e}")
            return []
            
        return instructions
    
    def _encode_instruction(self, instr: Instruction) -> int:
        """Encode an instruction into its binary representation based on the ISA documentation."""
        instr_def = self.instructions[instr.mnemonic]
        
        # Start with the opcode in the 5 most significant bits
        binary = (instr_def.opcode & 0b11111) << 27
        
        # Handle different instruction formats
        if instr_def.format == Format.BRANCH:
            if instr.mnemonic in ["beq", "bgt", "b", "call"]:
                if len(instr.operands) < 1:
                    raise SyntaxError(f"Line {instr.line_num}: Branch instruction requires a target")
                
                target_label = instr.operands[0].value
                if target_label not in self.symbol_table:
                    raise SyntaxError(f"Line {instr.line_num}: Undefined label '{target_label}'")
                
                target_addr = self.symbol_table[target_label]
                # PC-relative offset, divided by 4 (word-aligned)
                # PC is address of next instruction (current + 4)
                offset = (target_addr - (instr.address + 4)) >> 2
                
                # Encode the offset in the lower 27 bits
                # If negative, ensure proper 27-bit 2's complement representation
                if offset < 0:
                    offset = offset & 0x7FFFFFF  # Mask to 27 bits
                
                binary |= (offset & 0x7FFFFFF)
            
            # For nop and ret, opcode is sufficient
        
        elif instr_def.format == Format.REGISTER:
            # Check operand count
            if len(instr.operands) < 2:  # At minimum need 2 operands for reg-reg ops
                raise SyntaxError(f"Line {instr.line_num}: Register format instruction requires at least 2 operands")
            
            # For 3-address instructions like add rd, rs1, rs2
            if instr_def.operands == 3:
                # First operand is destination register (rd)
                if instr.operands[0].type != TokenType.REGISTER:
                    raise AssemblerError("First operand must be a register", instr.line_num)
                rd = int(instr.operands[0].value)
                self.validate_register(rd, instr.line_num)
                binary |= ((rd & 0xF) << 22)  # rd at bits 23-26
                
                # Second operand is first source register (rs1)
                if instr.operands[1].type != TokenType.REGISTER:
                    raise AssemblerError("Second operand must be a register", instr.line_num)
                rs1 = int(instr.operands[1].value)
                self.validate_register(rs1, instr.line_num)
                binary |= ((rs1 & 0xF) << 18)  # rs1 at bits 19-22
                
                # Third operand is either second source register (rs2) or immediate (imm)
                if instr.operands[2].type == TokenType.REGISTER:
                    # Register format - third operand is rs2
                    rs2 = int(instr.operands[2].value)
                    self.validate_register(rs2, instr.line_num)
                    binary |= ((rs2 & 0xF) << 14)  # rs2 at bits 15-18
                elif instr.operands[2].type == TokenType.IMMEDIATE:
                    # Immediate format - third operand is immediate
                    binary |= (1 << 26)  # Set I-bit to 1 for immediate
                    
                    # Parse immediate value
                    if instr.operands[2].value.startswith('0x'):
                        imm_value = int(instr.operands[2].value, 16)
                    else:
                        imm_value = int(instr.operands[2].value)
                    
                    # Validate and encode immediate value
                    self.validate_immediate(imm_value, signed=(imm_value < 0), line_num=instr.line_num)
                    
                    # Determine modifier bits (bits 16-17)
                    if imm_value < 0:
                        modifier = 0b00  # Signed immediate
                    else:
                        modifier = 0b01  # Unsigned immediate
                    
                    binary |= ((modifier & 0b11) << 16)
                    binary |= (imm_value & 0xFFFF)  # Lower 16 bits for immediate
            
            # For 2-address instructions like cmp rs1, rs2/imm or not rd, rs2/imm
            elif instr_def.operands == 2:
                if instr.mnemonic == "cmp":
                    # cmp rs1, rs2/imm
                    # First operand is rs1
                    if instr.operands[0].type != TokenType.REGISTER:
                        raise AssemblerError("First operand must be a register", instr.line_num)
                    rs1 = int(instr.operands[0].value)
                    self.validate_register(rs1, instr.line_num)
                    binary |= ((rs1 & 0xF) << 18)  # rs1 at bits 19-22
                    
                    # Second operand is rs2 or immediate
                    if instr.operands[1].type == TokenType.REGISTER:
                        rs2 = int(instr.operands[1].value)
                        self.validate_register(rs2, instr.line_num)
                        binary |= ((rs2 & 0xF) << 14)  # rs2 at bits 15-18
                    else:
                        # It's an immediate
                        binary |= (1 << 26)  # Set I-bit to 1
                        
                        # Parse immediate value
                        if instr.operands[1].value.startswith('0x'):
                            imm_value = int(instr.operands[1].value, 16)
                        else:
                            imm_value = int(instr.operands[1].value)
                        
                        # Validate immediate value
                        self.validate_immediate(imm_value, signed=(imm_value < 0), line_num=instr.line_num)
                        
                        # Determine modifier bits
                        if imm_value < 0:
                            modifier = 0b00  # Signed immediate
                        else:
                            modifier = 0b01  # Unsigned immediate
                        
                        binary |= ((modifier & 0b11) << 16)
                        binary |= (imm_value & 0xFFFF)  # Lower 16 bits for immediate
                
                elif instr.mnemonic == "not":
                    # not rd, rs2/imm
                    # First operand is rd
                    if instr.operands[0].type != TokenType.REGISTER:
                        raise AssemblerError("First operand must be a register", instr.line_num)
                    rd = int(instr.operands[0].value)
                    self.validate_register(rd, instr.line_num)
                    binary |= ((rd & 0xF) << 22)  # rd at bits 23-26
                    
                    # Second operand is rs2 or immediate
                    if instr.operands[1].type == TokenType.REGISTER:
                        rs2 = int(instr.operands[1].value)
                        self.validate_register(rs2, instr.line_num)
                        binary |= ((rs2 & 0xF) << 14)  # rs2 at bits 15-18
                    else:
                        # It's an immediate
                        binary |= (1 << 26)  # Set I-bit to 1
                        
                        # Parse immediate value
                        if instr.operands[1].value.startswith('0x'):
                            imm_value = int(instr.operands[1].value, 16)
                        else:
                            imm_value = int(instr.operands[1].value)
                        
                        # Validate immediate value
                        self.validate_immediate(imm_value, signed=(imm_value < 0), line_num=instr.line_num)
                        
                        # Determine modifier bits
                        if imm_value < 0:
                            modifier = 0b00  # Signed immediate
                        else:
                            modifier = 0b01  # Unsigned immediate
                        
                        binary |= ((modifier & 0b11) << 16)
                        binary |= (imm_value & 0xFFFF)  # Lower 16 bits for immediate
        
        elif instr_def.format == Format.IMMEDIATE:
            # Set I-bit to 1 for immediate format
            binary |= (1 << 26)
            
            if instr.mnemonic == "mov":
                # mov rd, imm/rs2
                # First operand is destination register (rd)
                if instr.operands[0].type != TokenType.REGISTER:
                    raise SyntaxError(f"Line {instr.line_num}: First operand must be a register")
                rd = int(instr.operands[0].value)
                binary |= ((rd & 0xF) << 22)  # rd at bits 23-26
                
                # Second operand is immediate or register
                if instr.operands[1].type == TokenType.REGISTER:
                    # If it's a register, we're actually using register format
                    binary &= ~(1 << 26)  # Clear I-bit
                    rs2 = int(instr.operands[1].value)
                    binary |= ((rs2 & 0xF) << 14)  # rs2 at bits 15-18
                else:
                    # It's an immediate
                    # Parse immediate value
                    if instr.operands[1].value.startswith('0x'):
                        imm_value = int(instr.operands[1].value, 16)
                        # Check if this is a high immediate (for movh)
                        if imm_value > 0xFFFF:
                            modifier = 0b10  # High immediate
                            imm_value = (imm_value >> 16) & 0xFFFF
                        else:
                            modifier = 0b01  # Unsigned immediate
                    else:
                        imm_value = int(instr.operands[1].value)
                        # Determine modifier bits
                        if imm_value < 0:
                            modifier = 0b00  # Signed immediate
                        else:
                            modifier = 0b01  # Unsigned immediate
                    
                    binary |= ((modifier & 0b11) << 16)
                    binary |= (imm_value & 0xFFFF)  # Lower 16 bits for immediate
            
            elif instr.mnemonic in ["ld", "st"]:
                # Memory operations: ld rd, imm[rs1] or st rd, imm[rs1]
                if len(instr.operands) < 3:
                    raise SyntaxError(f"Line {instr.line_num}: Memory instructions require a register and a memory reference")
                
                # First operand is register (rd)
                if instr.operands[0].type != TokenType.REGISTER:
                    raise SyntaxError(f"Line {instr.line_num}: First operand must be a register")
                rd = int(instr.operands[0].value)
                binary |= ((rd & 0xF) << 22)  # rd at bits 23-26
                
                # Second token should be immediate offset
                if instr.operands[1].type != TokenType.IMMEDIATE:
                    raise SyntaxError(f"Line {instr.line_num}: Expected immediate offset in memory reference")
                
                # Parse immediate value
                if instr.operands[1].value.startswith('0x'):
                    offset = int(instr.operands[1].value, 16)
                else:
                    offset = int(instr.operands[1].value)
                
                # Third token should be base register
                if instr.operands[2].type != TokenType.REGISTER:
                    raise SyntaxError(f"Line {instr.line_num}: Expected register in memory reference")
                rs1 = int(instr.operands[2].value)
                binary |= ((rs1 & 0xF) << 18)  # rs1 at bits 19-22
                
                # Determine modifier bits for offset
                modifier = 0b00 if offset < 0 else 0b01  # 00=signed, 01=unsigned
                binary |= ((modifier & 0b11) << 16)
                binary |= (offset & 0xFFFF)  # Lower 16 bits for offset
        
        # Ensure only 32-bit values are used
        binary &= 0xFFFFFFFF
        return binary
    
    def assemble(self, input_file: str, output_file: str) -> None:
        """Assemble the input file and write the binary to the output file."""
        # Reset state
        self.symbol_table = {}
        self.current_address = 0
        self.binary_output = []
        
        try:
            # First pass: Parse file and build symbol table
            print(f"Assembling {input_file}...")
            instructions = self._parse_file(input_file)
            
            if not instructions:
                print("No instructions to assemble!")
                return
            
            # Second pass: Encode instructions
            print("\nEncoding instructions:")
            for instr in instructions:
                try:
                    binary = self._encode_instruction(instr)
                    self.binary_output.append(binary)
                    
                    # Print results to terminal
                    print(f"0x{instr.address:08X}: {instr.mnemonic} → 0x{binary:08X} (bin: {binary:032b})")
                    
                except Exception as e:
                    print(f"Error at line {instr.line_num}: {str(e)}")
                    return
            
            # Write binary and hex output
            bin_filename = output_file
            hex_filename = output_file.replace('.bin', '.hex')
            
            try:
                with open(bin_filename, 'wb') as bin_f, open(hex_filename, 'w') as hex_f:
                    for i, binary in enumerate(self.binary_output):
                        bin_f.write(struct.pack('<I', binary))  # Little-endian 32-bit
                        hex_string = format(binary, '08X')
                        hex_f.write(f"Instruction {i}: 0x{hex_string}\n")
                
                print(f"\nAssembly completed successfully!")
                print(f"Binary output written to: {bin_filename}")
                print(f"Hex output written to: {hex_filename}")
                print(f"Total instructions: {len(self.binary_output)}")
                    
            except Exception as e:
                print(f"Error writing output files: {str(e)}")
                
        except AssemblerError as e:
            print(e.format_message())
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='SimpleRisc Assembler')
    parser.add_argument('input_file', help='Input assembly file')
    parser.add_argument('-o', '--output', help='Output binary file', default='output.bin')
    args = parser.parse_args()
    
    try:
        assembler = SimpleRiscAssembler()
        assembler.assemble(args.input_file, args.output)
    except Exception as e:
        print(f"Assembler error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()