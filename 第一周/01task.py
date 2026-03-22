import json
import numpy as np
from typing import Dict, List, Union, Tuple
import math

class CoordinateSystem:
    """坐标系类，处理向量的坐标系转换及相关计算"""
    
    def __init__(self, axes: List[List[float]], name: str = "current"):
        """
        初始化坐标系
        
        Args:
            axes: 坐标轴向量列表，每个轴是一个列表
            name: 坐标系名称
        
        Raises:
            ValueError: 如果不能构成有效的坐标系
        """
        self.axes = np.array(axes, dtype=float)
        self.name = name
        self.dimension = len(axes)
        
        # 检查是否能构成有效坐标系
        self._validate_coordinate_system()
        
        # 计算转换矩阵
        self.transform_matrix = self._compute_transform_matrix()
        
    def _validate_coordinate_system(self):
        """验证坐标轴是否能构成有效的坐标系"""
        
        # 检查维度一致性
        for i, axis in enumerate(self.axes):
            if len(axis) != self.dimension:
                raise ValueError(f"轴 {i} 的维度 ({len(axis)}) 与坐标系维度 ({self.dimension}) 不一致")
        
        # 检查线性无关性
        rank = np.linalg.matrix_rank(self.axes)
        if rank < self.dimension:
            raise ValueError(f"坐标轴线性相关，无法构成 {self.dimension} 维坐标系 (秩={rank})")
        
        # 检查是否存在零向量
        for i, axis in enumerate(self.axes):
            if np.allclose(axis, 0):
                raise ValueError(f"轴 {i} 是零向量，无效的坐标轴")
    
    def _compute_transform_matrix(self) -> np.ndarray:
        """
        计算从当前坐标系到标准正交基的转换矩阵
        返回: 转换矩阵
        """
        # 对于一般坐标系，我们需要解线性方程组
        # 假设标准正交基下的向量 e 可以通过坐标轴线性组合得到
        # 即 v_standard = axes @ v_local
        
        # 对于从标准正交基到当前坐标系的转换
        to_standard = self.axes.T
        
        # 从当前坐标系到标准正交基的转换（逆变换）
        try:
            from_standard = np.linalg.inv(to_standard)
        except np.linalg.LinAlgError:
            raise ValueError("坐标轴矩阵不可逆，无法进行坐标转换")
        
        return from_standard
    
    def to_standard_basis(self, vector: np.ndarray) -> np.ndarray:
        """
        将向量从当前坐标系转换到标准正交基
        
        Args:
            vector: 在当前坐标系下的坐标
            
        Returns:
            在标准正交基下的坐标
        """
        vector = np.array(vector, dtype=float)
        if len(vector) != self.dimension:
            raise ValueError(f"向量维度 ({len(vector)}) 与坐标系维度 ({self.dimension}) 不一致")
        
        # v_standard = axes @ v_local
        return self.axes.T @ vector
    
    def from_standard_basis(self, vector: np.ndarray) -> np.ndarray:
        """
        将向量从标准正交基转换到当前坐标系
        
        Args:
            vector: 在标准正交基下的坐标
            
        Returns:
            在当前坐标系下的坐标
        """
        vector = np.array(vector, dtype=float)
        if len(vector) != self.dimension:
            raise ValueError(f"向量维度 ({len(vector)}) 与坐标系维度 ({self.dimension}) 不一致")
        
        return self.transform_matrix @ vector
    
    def transform_to(self, vector: np.ndarray, target_coord: 'CoordinateSystem') -> np.ndarray:
        """
        将向量从当前坐标系转换到目标坐标系
        
        Args:
            vector: 在当前坐标系下的向量
            target_coord: 目标坐标系
            
        Returns:
            在目标坐标系下的向量表示
        """
        # 1. 将当前向量转换到标准正交基
        v_standard = self.to_standard_basis(vector)
        
        # 2. 从标准正交基转换到目标坐标系
        return target_coord.from_standard_basis(v_standard)
    
    def project(self, vector: np.ndarray) -> List[float]:
        """
        计算向量在每个轴上的投影长度
        
        Args:
            vector: 在当前坐标系下的向量
            
        Returns:
            在每个轴上的投影长度列表
        """
        vector = np.array(vector, dtype=float)
        projections = []
        
        for axis in self.axes:
            # 投影长度 = |v| * cos(theta) = (v·axis) / |axis|
            proj = np.dot(vector, axis) / np.linalg.norm(axis)
            projections.append(float(proj))
        
        return projections
    
    def angles(self, vector: np.ndarray) -> List[float]:
        """
        计算向量与每个轴的夹角弧度
        
        Args:
            vector: 在当前坐标系下的向量
            
        Returns:
            与每个轴的夹角弧度列表
        """
        vector = np.array(vector, dtype=float)
        v_norm = np.linalg.norm(vector)
        
        if v_norm == 0:
            return [0.0] * self.dimension
        
        angles = []
        for axis in self.axes:
            axis_norm = np.linalg.norm(axis)
            cos_theta = np.dot(vector, axis) / (v_norm * axis_norm)
            # 处理数值误差
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angles.append(float(math.acos(cos_theta)))
        
        return angles
    
    def area_scale(self) -> float:
        """
        计算相对于标准直角坐标系的面积/体积缩放倍数
        
        Returns:
            缩放倍数
        """
        # 对于二维：面积缩放 = |det(axes)|
        # 对于三维：体积缩放 = |det(axes)|
        # 对于n维：n维体积缩放 = |det(axes)|
        return float(abs(np.linalg.det(self.axes)))
    
    def __str__(self):
        return f"坐标系 '{self.name}' (维度={self.dimension})"


class VectorProcessor:
    """向量处理器，处理JSON任务"""
    
    def __init__(self, json_data: Dict):
        """
        初始化处理器
        
        Args:
            json_data: 包含原始向量、坐标轴和任务列表的JSON数据
        """
        self.original_vector = np.array(json_data.get('vector', []), dtype=float)
        self.current_coord = CoordinateSystem(
            json_data.get('axes', []), 
            name="初始坐标系"
        )
        self.tasks = json_data.get('tasks', [])
        
        print(f"初始化完成:")
        print(f"原始向量: {self.original_vector}")
        print(f"初始坐标系: {self.current_coord}")
        print(f"坐标轴矩阵:\n{self.current_coord.axes}")
        print()
    
    def execute_tasks(self):
        """执行所有任务"""
        current_coord = self.current_coord
        current_vector = self.original_vector.copy()
        
        for i, task in enumerate(self.tasks):
            print(f"\n{'='*50}")
            print(f"执行任务 {i+1}: {task}")
            
            task_type = task.get('type')
            
            if task_type == '坐标系转移':
                current_coord, current_vector = self._handle_transfer(task, current_coord, current_vector)
            
            elif task_type == '坐标系投影':
                self._handle_projection(task, current_coord, current_vector)
            
            elif task_type == '坐标系夹角':
                self._handle_angles(task, current_coord, current_vector)
            
            elif task_type == '坐标系面积':
                self._handle_area(task, current_coord)
            
            else:
                print(f"未知的任务类型: {task_type}")
    
    def _handle_transfer(self, task: Dict, current_coord: CoordinateSystem, 
                        current_vector: np.ndarray) -> Tuple[CoordinateSystem, np.ndarray]:
        """处理坐标系转移任务"""
        try:
            # 创建目标坐标系
            target_axes = task.get('target_axes', [])
            target_coord = CoordinateSystem(target_axes, name="目标坐标系")
            
            print(f"目标坐标系轴:\n{target_coord.axes}")
            
            # 转换向量
            new_vector = current_coord.transform_to(current_vector, target_coord)
            
            print(f"转换前的向量 (在{current_coord}): {current_vector}")
            print(f"转换后的向量 (在{target_coord}): {new_vector}")
            
            # 验证：转换回原坐标系应该得到原始向量
            verified = target_coord.transform_to(new_vector, current_coord)
            print(f"验证转换回原坐标系: {verified}")
            print(f"验证结果: {'✓ 正确' if np.allclose(verified, current_vector) else '✗ 错误'}")
            
            return target_coord, new_vector
            
        except Exception as e:
            print(f"坐标系转移失败: {e}")
            return current_coord, current_vector
    
    def _handle_projection(self, task: Dict, current_coord: CoordinateSystem, 
                          current_vector: np.ndarray):
        """处理投影任务"""
        try:
            projections = current_coord.project(current_vector)
            
            print(f"向量 {current_vector} 在各轴上的投影:")
            for i, proj in enumerate(projections):
                print(f"  轴 {i+1}: {proj:.6f}")
                
        except Exception as e:
            print(f"投影计算失败: {e}")
    
    def _handle_angles(self, task: Dict, current_coord: CoordinateSystem, 
                      current_vector: np.ndarray):
        """处理夹角任务"""
        try:
            angles = current_coord.angles(current_vector)
            angles_deg = [math.degrees(a) for a in angles]
            
            print(f"向量 {current_vector} 与各轴的夹角:")
            for i, (rad, deg) in enumerate(zip(angles, angles_deg)):
                print(f"  轴 {i+1}: {rad:.6f} 弧度 ({deg:.2f}°)")
                
        except Exception as e:
            print(f"夹角计算失败: {e}")
    
    def _handle_area(self, task: Dict, current_coord: CoordinateSystem):
        """处理面积缩放任务"""
        try:
            scale = current_coord.area_scale()
            unit = "体积" if current_coord.dimension == 3 else "面积" if current_coord.dimension == 2 else f"{current_coord.dimension}维体积"
            
            print(f"相对于直角坐标系，{unit}缩放倍数为: {scale:.6f}")
            
        except Exception as e:
            print(f"面积计算失败: {e}")


def create_example_json():
    """创建一个示例JSON数据"""
    return {
        "vector": [1, 1],
        "axes": [[1, 0], [0, 1]],  # 标准直角坐标系
        "tasks": [
            {
                "type": "坐标系转移",
                "target_axes": [[1, 0], [1, 1]]
            },
            {
                "type": "坐标系投影",
            },
            {
                "type": "坐标系夹角",
            },
            {
                "type": "坐标系面积",
            }
        ]
    }


def create_3d_example():
    """创建一个3D示例"""
    return {
        "vector": [1, 1, 1],
        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 标准直角坐标系
        "tasks": [
            {
                "type": "坐标系转移",
                "target_axes": [
                    [1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 3]
                ]
            },
            {
                "type": "坐标系投影",
            },
            {
                "type": "坐标系夹角",
            },
            {
                "type": "坐标系面积",
            }
        ]
    }


def create_invalid_example():
    """创建一个无效坐标系示例（用于测试错误处理）"""
    return {
        "vector": [1, 1],
        "axes": [[1, 0], [2, 0]],  # 线性相关的轴，无法构成坐标系
        "tasks": [
            {
                "type": "坐标系转移",
                "target_axes": [[1, 0], [0, 1]]
            }
        ]
    }


def main():
    """主函数"""
    print("向量坐标系转换系统")
    print("="*60)
    
    # 示例1：标准2D示例
    print("\n\n示例1：2D坐标系转换")
    print("-"*40)
    data1 = create_example_json()
    processor1 = VectorProcessor(data1)
    processor1.execute_tasks()
    
    # 示例2：3D示例
    print("\n\n示例2：3D坐标系转换")
    print("-"*40)
    data2 = create_3d_example()
    processor2 = VectorProcessor(data2)
    processor2.execute_tasks()
    
    # 示例3：错误处理示例
    print("\n\n示例3：无效坐标系测试")
    print("-"*40)
    try:
        data3 = create_invalid_example()
        processor3 = VectorProcessor(data3)
    except ValueError as e:
        print(f"成功捕获错误: {e}")
    
    # 示例4：自定义JSON
    print("\n\n示例4：自定义JSON输入")
    print("-"*40)
    custom_json = input("是否输入自定义JSON? (y/n): ").lower()
    
    if custom_json == 'y':
        print("请输入JSON字符串:")
        json_str = input()
        try:
            data4 = json.loads(json_str)
            processor4 = VectorProcessor(data4)
            processor4.execute_tasks()
        except json.JSONDecodeError:
            print("JSON格式错误")
        except Exception as e:
            print(f"处理出错: {e}")
    else:
        print("跳过自定义输入")


if __name__ == "__main__":
    main()
