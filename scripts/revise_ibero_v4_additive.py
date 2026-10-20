"""V4 纠偏：完整保留 V3 的每一页，仅在仿真器章节之后插入精炼补充页。"""
from pathlib import Path
import hashlib
import json
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

import build_ibero_presentation_v4 as d


OUT = d.ROOT / "artifacts/presentation_v4_additive"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    backup = OUT / "IBERO_v4_上次改写版_备份.pptx"
    if not backup.exists():
        shutil.copy2(d.OUTPUT, backup)
    source_sha = digest(d.SOURCE.read_bytes())
    s = d.new_slide("Core 实现概览：菜谱、物理与材料闭环", "补充 / CORE 实现", subtitle="以 MuJoCo 为求解底座，把场景参数、接触响应与任务判定连接起来。")
    d.card(s, "场景菜谱", "scene_config.yaml：对象与物性\nconstraints.yaml：约束与阈值\ntask_spec.py：阶段与终止", 0.66, 2.43, 3.80, 2.60, d.BLUE, 15)
    d.card(s, "仿真 Core", "G1 上半身 + 可配置末端工具\n控制命令 → 接触 / 动力学积分\n读取力、位姿与材料状态\n记录任务结果和审计轨迹", 4.76, 2.43, 3.80, 2.60, d.TEAL, 15)
    d.card(s, "材料与接触", "线束：一维 Flex + 弯曲力\n卡扣：被动弹性梁 + 接触\n铣削：体素去除 + 过程载荷\n物性由场景配置，不写死在任务中", 8.86, 2.43, 3.80, 2.60, d.GOLD, 15)
    d.flow(s, [("P4 菜谱", "对象 / 参数 / 目标"), ("P2 + P3 组件", "机器人 / 工具 / 材料"), ("MuJoCo 求解", "接触 / 运动 / 受力"), ("任务与审计", "结果 / 越界 / 回放")], 5.42, w=2.62, gap=0.50)
    d.notes(s, "这是新增补充页，不改写原方案或原有接口页。当前使用 MuJoCo 3.8.1 与 G1 仿真模型。线束是无扭转中心线 Flex，轴向弹簧阻尼由引擎求解，弯曲力来自能量负梯度；端点与端头的材料连接不是夹爪抓取捷径，夹爪仍通过接触夹持。原有PPT全页保持不变。")
    d.video_slide("01_cable_handover", "Demo 01｜双臂线束端头交接与放置", "CABLE HANDOVER", ["接收夹爪建立接触夹持", "原夹爪释放，支撑关系转移", "末端送到接料区并放开"], "刚性端头交接，不宣称裸线束抓取。", "Flex 线束 + 夹爪真实接触；任务读取支撑与目标距离。", "重点看两手释放先后与最终工件支撑。")
    d.video_slide("02_cable_stretch", "Demo 02｜双臂拉伸与张力反馈", "CABLE TENSION", ["两端夹持使线束由松弛变紧", "控制器读取实际材料张力", "在目标范围内保持并接受审计"], "目标阈值是任务约束，不是实测破坏载荷。", "轴向边弹簧/阻尼 + 弯曲力；控制命令改变实际运动。", "观察线束形状和张力的变化。")
    s = d.new_slide("卡扣实现：接触按压 → 弹性变形 → 几何解锁", "补充 / 弹性材料与机构", subtitle="扣齿与肩部的真实接触阻挡拔出；按压产生净空后，插头才能退出。")
    d.flow(s, [("未按压", "扣齿—肩部接触阻挡"), ("工具按压", "接触力弯曲舌片"), ("形成净空", "实际几何越过肩部"), ("退出与回弹", "拉出 / 松开 / 弹性恢复")], 2.47, w=2.62, gap=0.50)
    d.card(s, "材料怎么实现", "分段刚体 + 被动转角弹簧 / 阻尼\n刚度由 E、截面 I 与段长 h 决定\n普通段 kθ = EI/h；不是预设变形动画", 0.66, 4.04, 7.18, 2.09, d.TEAL, 17)
    d.card(s, "任务怎么判定", "读取净空、退出距离与接触力\n状态标签只观察物理结果\n不靠关闭碰撞或删除锁止约束", 8.14, 4.04, 4.52, 2.09, d.BLUE, 16)
    d.ribbon(s, "当前为弹性梁近似与示例物性；尚不包含塑性、疲劳及真实断裂预测。")
    d.notes(s, "源码：materials/elastic_beam.py、mechanisms/snap_latch.py。卡扣分段梁为被动弹性元件，根部半单元刚度2EI/h；净空变化由接触力作用产生。用于解释随后两段卡扣与带线束拔出视频。")
    d.video_slide("03_latch_release", "Demo 03｜按压弹性卡扣并拔出", "LATCH RELEASE", ["按压工具接触并压弯舌片", "净空由负变正，再开始退出", "撤离后舌片回弹，插头已分离"], "示例材料参数，未标定具体商品连接器。", "弹性梁 + 扣齿/肩部碰撞；夹持依赖真实接触。", "结合净空和接触力曲线讲解按压与拔出的先后因果。")
    d.video_slide("04_harness_unplug", "Demo 04｜带线束的解锁、拔出与放置", "HARNESS UNPLUG", ["复用同一个卡扣与按压机制", "线束随插头运动并产生张力", "放到接料盘后释放夹爪"], "接料盘投放，不宣称无落差精密装配。", "卡扣 + Flex 线束 + 任务链；端头质量只计一次。", "展示卡扣和线束组件组合形成任务链。")
    s = d.new_slide("铣削实现：材料去除与切削载荷同步反馈", "补充 / 加工机制", subtitle="固定体素工件记录材料占据；刀具扫掠更新孔槽，过程载荷反馈到机器人与主轴。")
    d.flow(s, [("实际刀具运动", "位姿 / 进给 / 转速"), ("计算过程载荷", "啮合 + 切削系数"), ("提交材料去除", "体素 / 碰撞 / 事件"), ("动力学反馈", "力 / 扭矩 → 运动")], 2.50, w=2.62, gap=0.50)
    d.card(s, "孔槽不是后期画出来的", "刀具扫掠确定被去除的体素。\n同一事件更新材料账本、碰撞与显示状态。\n俯视、剖面和形状检查共同核对加工结果。", 0.66, 4.02, 7.20, 2.14, d.TEAL, 17)
    d.card(s, "载荷进入物理闭环", "切削力 / 扭矩进入统一载荷汇总\n机器人和主轴运动受载荷影响\n超限工况由审计记录并判失败", 8.16, 4.02, 4.50, 2.14, d.BLUE, 16)
    d.ribbon(s, "过程级近似模型，飞屑仅可视化；当前左臂加工、右臂停放。")
    d.notes(s, "源码：materials/stock.py、stock_collision.py、processes/milling.py、core/milling_runtime.py。使用三轴平均载荷模型与固定体素工件，不是金属塑性断裂FEM。后续三个视频分别展示贯通孔、直槽与浅腔，源轨迹与版本信息见对应备注。")
    d.video_slide("05_milling_through_hole", "Demo 05｜铣削通孔：从实体到贯通", "THROUGH HOLE", ["主轴启动，工具向实体进给", "观察材料剖面逐层去除", "贯通后保留孔形与载荷记录"], "飞屑仅可视化；右臂安全停放。", "真实主轴/机器人运动 + 体素碰撞去除 + 过程载荷。", "对照俯视与剖面观察孔形成。")
    d.video_slide("06_milling_slot", "Demo 06｜铣削直槽：路径改变加工形状", "SLOT MILLING", ["先下刀，再沿槽方向进给", "扫掠去除形成连续槽形", "观察转速、载荷与材料体积"], "当前展示浅槽，不把它标成贯通孔。", "复用相同材料/加工内核，改变目标和控制路径。", "关注下刀与水平进给两个阶段。")
    d.video_slide("07_milling_pocket", "Demo 07｜铣削浅腔：二维区域逐步去除", "POCKET MILLING", ["连续轨迹逐步扩大去除区域", "浅腔底部仍保留实体材料", "停止主轴后核对最终腔形"], "示例过程系数，未宣称真实钢材切削精度。", "材料去除事件 + 形状指标 + 物理状态轨迹。", "用剖面区分浅腔与贯通孔。")
    for i, slide in enumerate(d.order, 1):
        d.text(slide, f"新增 {i:02d}", 11.80, 7.13, 0.85, 0.23, 9, d.MUTED)
    temporary = OUT / "additions_package.pptx"
    # 不重排 python-pptx 中的原页，确保其原部件路径仍是 slide1..19。
    d.prs.save(temporary)
    with ZipFile(d.SOURCE) as original, ZipFile(temporary) as generated:
        original_names = set(original.namelist())
        presentation = ET.fromstring(generated.read("ppt/presentation.xml"))
        ids = presentation.find(f"{{{P}}}sldIdLst")
        values = list(ids)
        assert len(values) == 29
        for value in values:
            ids.remove(value)
        for value in values[:9] + values[19:] + values[9:19]:
            ids.append(value)
        # 只有包级目录/关系加入新页；V3 原页、备注、主题、母版逐字节复制。
        replacements = {
            "ppt/presentation.xml": ET.tostring(presentation, encoding="utf-8", xml_declaration=True),
            "ppt/_rels/presentation.xml.rels": generated.read("ppt/_rels/presentation.xml.rels"),
            "[Content_Types].xml": generated.read("[Content_Types].xml"),
        }
        staged = OUT / "V4_已核对原页.pptx"
        with ZipFile(staged, "w", ZIP_DEFLATED) as output:
            for item in original.infolist():
                output.writestr(item, replacements.get(item.filename, original.read(item.filename)))
            for item in generated.infolist():
                if item.filename not in original_names:
                    output.writestr(item, generated.read(item.filename))
    with ZipFile(d.SOURCE) as original, ZipFile(staged) as output:
        preserved = [name for name in original.namelist() if name not in replacements]
        assert all(original.read(name) == output.read(name) for name in preserved)
        assert output.testzip() is None
        videos = [n for n in output.namelist() if n.startswith("ppt/media/") and n.endswith(".mp4")]
        assert len(videos) == 7
        assert {digest(output.read(n)) for n in videos} == {row["sha256"] for row in d.entries if row["kind"] == "video"}
    assert digest(d.SOURCE.read_bytes()) == source_sha
    shutil.copy2(staged, d.OUTPUT)
    manifest = {
        "mode": "additions_only", "slides": 29, "original_slides_preserved": 19,
        "new_logic_slides": 3, "embedded_videos": 7, "insert_after_original_slide": 9,
        "source_v3_sha256": source_sha, "output_sha256": digest(d.OUTPUT.read_bytes()),
        "original_parts_preserved_byte_for_byte": len(preserved),
        "original_slide_positions": list(range(1, 10)) + list(range(20, 30)),
        "video_positions": [11, 12, 14, 15, 17, 18, 19],
        "previous_v4_backup": backup.name,
        "additions": [dict(row, slide=row["slide"] + 9) for row in d.entries],
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
