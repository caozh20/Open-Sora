"""
Patches to fix HybridParallelPlugin issues in ColossalAI
"""
import torch
import sys
import types
from functools import wraps
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.booster import Booster

# 保存原始的 __del__ 方法
original_del = HybridParallelPlugin.__del__

# 创建一个处理异常的 __del__ 方法
def safe_del(self):
    """安全的资源释放，处理可能的属性缺失情况"""
    try:
        if hasattr(self, 'pg_mesh'):
            self.pg_mesh.destroy_mesh_process_groups()
    except Exception as e:
        # 忽略异常，但可以打印调试信息
        # print(f"Warning: Error during HybridParallelPlugin cleanup: {e}", file=sys.stderr)
        pass

# 创建一个安全的 Booster.boost 方法
original_boost = Booster.boost

@wraps(original_boost)
def safe_boost(self, *args, **kwargs):
    """包装原始boost方法，确保pg_mesh属性存在"""
    result = original_boost(self, *args, **kwargs)
    
    # 确保插件有pg_mesh属性
    if hasattr(self, 'plugin') and isinstance(self.plugin, HybridParallelPlugin):
        if not hasattr(self.plugin, 'pg_mesh'):
            # 创建一个空的对象作为pg_mesh
            class DummyPGMesh:
                def destroy_mesh_process_groups(self):
                    pass
            self.plugin.pg_mesh = DummyPGMesh()
    
    return result

# 应用补丁
def apply_hybrid_plugin_patches():
    """应用所有补丁来修复HybridParallelPlugin的问题"""
    # 替换 __del__ 方法
    HybridParallelPlugin.__del__ = safe_del
    
    # 替换 boost 方法
    Booster.boost = safe_boost
    
    return True

# 自动应用补丁
patch_applied = apply_hybrid_plugin_patches() 