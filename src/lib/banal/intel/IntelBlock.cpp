#include "IntelBlock.hpp"
#include <Instruction.h>


namespace Dyninst {
namespace ParseAPI {

IntelBlock::IntelBlock(CodeObject * o, CodeRegion * r,
  Address start, std::vector<Offset> &offsets) : Block(o, r, start) {
  for (auto offset : offsets) {
    _inst_offsets.push_back(offset);
  }
}


Address IntelBlock::last() const {
  return this->_inst_offsets.back();
}


void IntelBlock::getInsns(Insns &insns) const {
  for (auto offset : _inst_offsets) {
#ifdef DYNINST_INSTRUCTION_PTR
    insns.insert(std::pair<long unsigned int, 
      InstructionAPI::InstructionPtr>(offset, NULL));
#else
    InstructionAPI::Instruction inst;    
    insns[offset] = inst;
#endif
  }
}

}
}
