"""New episode asset: explicitly initialized powder and estimated top-handle rocker."""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
import mujoco
out=Path('results/brew-scene-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
t=ET.parse('results/prepared-layout-007/scene.xml');root=t.getroot();base=t.find('.//body[@name="base_eletrica"]')
for x in list(base):
 if x.get('name') in ['botao_chaleira','botao_chaleira_site']:base.remove(x)
jar=t.find('.//body[@name="chaleira"]');button=ET.SubElement(jar,'body',name='kettle_rocker',pos='.100 0 .205');ET.SubElement(button,'joint',name='kettle_rocker_hinge',type='hinge',axis='0 1 0',range='0 .21',limited='true',stiffness='.3',damping='.006',frictionloss='.002');ET.SubElement(button,'geom',name='kettle_rocker_contact',type='box',pos='.008 0 0',size='.012 .009 .003',mass='.005',rgba='.025 .025 .03 1',friction='1 .005 .0001');ET.SubElement(button,'site',name='kettle_rocker_tip',pos='.018 0 .003',size='.002',rgba='.8 .1 .1 .6')
for name,geom in [('pote',dict(name='grounds_pot_visual',type='cylinder',pos='0 0 .0268',size='.051 .0168',rgba='.12 .055 .02 1')),('scoop',dict(name='grounds_spoon_visual',type='ellipsoid',pos='-.045 0 .003',size='.01 .008 .001',rgba='.12 .055 .02 0')),('coador',dict(name='grounds_filter_visual',type='ellipsoid',pos='0 0 .156',size='.022 .022 .008',rgba='.12 .055 .02 0'))]:
 ET.SubElement(t.find('.//body[@name="'+name+'"]'),'geom',**geom,contype='0',conaffinity='0',density='0',group='1')
custom=root.find('custom')
if custom is None:custom=ET.SubElement(root,'custom')
ET.SubElement(custom,'numeric',name='brew_initial_powder_g',data='100');ET.SubElement(custom,'numeric',name='brew_enabled',data='1')
scene=(out/'scene.xml').resolve();t.write(scene);m=mujoco.MjModel.from_xml_path(str(scene));r={'model':'gpt-6-astra','scene':str(scene),'compiled':True,'coffee_completed':False,'scope':'NEW episode initialization asset; no physical stage validated in this scene','initial_powder_g':100,'kettle_switch_location':'top of handle per official EEK10 manual page1, replacing wrong base proxy','rocker_geometry_assumed_mm':{'length':24,'width':18,'thickness':6},'rocker_dynamics_assumed':{'mass_kg':.005,'spring_Nm_rad':.3,'travel_rad':.21},'source_scene':'results/prepared-layout-007/scene.xml','limitations':['switch dimensions, spring and latching mechanics estimated; not measured EEK10','grounds reduced inventory/mass/visual model, not granular contacts','5g rocker added conservatively on top of780g body proxy']};(out/'report.json').write_text(json.dumps(r,indent=2));print(r)
