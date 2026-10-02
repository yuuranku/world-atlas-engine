/* Cultural building grammar and reusable landmark footprints, in local metres. */
(() => {
  'use strict';
  const styles = {
    courtyard: {roof:['#c4a08a','#d1b39a','#ba917c'],court:true,shrink:.12,temple:'cloister'},
    arcaded: {roof:['#c9ac85','#d8bf9b','#bfa17e'],court:true,shrink:.10,temple:'colonnade'},
    'walled-compound': {roof:['#aa9983','#c5b598','#b3a288'],court:true,shrink:.15,temple:'cloister'},
    'timber-frame': {roof:['#9d8170','#b49a83','#c2a68c'],court:false,shrink:.24,temple:'longhouse'},
    'stone-masonry': {roof:['#b9a48c','#c9b99e','#a69581'],court:false,shrink:.12,temple:'colonnade'},
    terraced: {roof:['#c6a68c','#d5b69b','#b5947c'],court:false,shrink:.06,temple:'longhouse'},
    'canal-side': {roof:['#b59880','#c9ac91','#a98973'],court:false,shrink:.08,temple:'longhouse'},
    'vernacular-mixed': {roof:['#ba9a82','#cab294','#ae917c'],court:false,shrink:.17,temple:'cloister'},
    nomadic: {roof:['#e4d7b7','#d2c39e','#c6b898'],court:false,shrink:.48,temple:'pavilion'},
  };
  const traditions={
    courtyard:{plan:'axial',streets:['槐荫','书院','文庙','钟楼','鼓楼','漕运','东市','西市','绸缎','瓷窑','药铺','南仓','柳桥','盐行','会馆','牌楼','织坊','青石','竹园','武庙'],hall:'府署',market:'市集',temple:'庙院'},
    'walled-compound':{plan:'axial',streets:['御道','承天','礼坊','司仓','学宫','武库','太平','青龙','朱雀','白虎','玄武','永安','东仓','水门','织锦','石坊','驿馆','钟鼓','通远','曲院'],hall:'官署',market:'东市',temple:'祠庙'},
    arcaded:{plan:'orthogonal',streets:['柱廊','议会','竞技','陶工','橄榄','浴场','剧院','学园','铜匠','凯旋','公仓','神殿','泉水','葡萄','商馆','石雕','谷物','染坊','长廊','船坞'],hall:'议政厅',market:'市集',temple:'神殿'},
    'timber-frame':{plan:'organic',streets:['磨坊','木匠','橡树','铁匠','羊毛','麦芽','酿酒','皮匠','面包','钟塔','修院','马厩','旧集','猎人','松林','渡口','北仓','磨石','牛市','小教堂'],hall:'市政厅',market:'集市',temple:'修院'},
    'stone-masonry':{plan:'organic',streets:['石桥','石匠','银匠','商会','教堂','钟塔','旧港','城堡','葡萄','泉井','议会','花园','染坊','谷市','烛坊','高台','盐仓','砌石','朝圣','修道院'],hall:'市政厅',market:'市集',temple:'圣堂'},
    terraced:{plan:'contour',streets:['上台','下台','石阶','观景','山泉','崖边','坡脚','梯田','樵夫','高仓','山门','瞭望','石径','茶园','岭脊','松风','望海','泉桥','采石','云台'],hall:'议事厅',market:'山集',temple:'山寺'},
    'canal-side':{plan:'waterfront',streets:['船坞','鱼市','帆匠','缆索','水门','盐仓','泊船','海关','桅杆','渔网','滨河','石堤','渡桥','河埠','旧港','南码头','帆布','造船','商馆','灯塔'],hall:'商议厅',market:'鱼市',temple:'水神庙'},
    'vernacular-mixed':{plan:'organic',streets:['榆树','匠人','长集','南仓','旧井','柳桥','织工','陶窑','铁坊','药草','客栈','粮市','青石','泉水','东园','驿馆','木坊','书舍','盐铺','新桥'],hall:'议事厅',market:'市集',temple:'神庙'},
    nomadic:{plan:'camp',streets:['牧场','驼铃','井泉','集会','草场','马市','羊市','商旅','帐庭','驿站'],hall:'议事帐',market:'马市',temple:'祭祀场'},
  };
  const materials={
    tile:{roof:['#ad6450','#bc7960','#925948','#c78a70','#a46c57'],palace:['#b66948','#c48553','#95553c']},
    slate:{roof:['#61777b','#70878a','#53676f','#82918d','#6b7377'],palace:['#536b74','#728b90','#435e68']},
    timber:{roof:['#82745b','#998367','#6d6251','#ab9372','#8b7059'],palace:['#756b53','#998369','#5e6251']},
    lime:{roof:['#d9c7a3','#c9b78f','#e4d4b7','#baa98d','#d0bea4'],palace:['#c3a873','#dec48c','#ae9467']},
    glazed:{roof:['#5d7973','#718d79','#485f62','#839684','#60746b'],palace:['#ad843b','#c6a152','#8e692f']},
  };
  function materialProfile(recipe,style){
    const latitude=Math.abs(recipe.siteContext?.latitudeDegrees||recipe.localSite?.latitudeDegrees||0);
    const id=recipe.culture?.civilizationId||0,variant=id%3;
    const name=['courtyard','walled-compound'].includes(style)?variant===1?'glazed':variant===2?'tile':'slate':
      style==='arcaded'?'lime':style==='timber-frame'?'timber':style==='terraced'||latitude>48?'slate':
      style==='nomadic'?'lime':variant===0?'tile':variant===1?'slate':'timber';
    return {name,...materials[name]};
  }
  function palaceDesign(recipe,style){
    const id=(recipe.seed^(recipe.culture?.civilizationId||0)*2654435761)>>>0;
    const rugged=['ridge','hillside','coastal-slope','valley'].includes(recipe.localSite?.form.kind);
    const family=rugged?'terraced-court':['courtyard','walled-compound'].includes(style)?'axial-courts':
      style==='arcaded'?'peristyle':style==='timber-frame'?'hall-and-bailey':
      style==='canal-side'?'merchant-palazzo':style==='stone-masonry'?'castle-palace':'garden-pavilions';
    return {family,variant:id%3,aspect:family==='terraced-court'?[1.25,.62]:family==='axial-courts'?[.93,1.28]:
      family==='merchant-palazzo'?[1.30,.76]:[1.28,1],fortified:['castle-palace','hall-and-bailey'].includes(family)};
  }
  function profile(recipe) {
    const culture=recipe.culture||{},population=recipe.population?.estimate||0,type=recipe.siteType;
    const mobile=['nomadic-camp','khan-court'].includes(type)||culture.mobility==='nomadic'
      ||culture.government==='nomadic-confederacy'&&recipe.tier==='site'&& !['port','island-port','lake-port','river-city','fortress','pass'].includes(type);
    const village=!mobile&&(type==='village'||population>0&&population<1800&&!['fortress','pass','port','island-port'].includes(type));
    const small=mobile||village||population<1800;
    const style=mobile?'nomadic':culture.style||'vernacular-mixed';
    const kind=mobile?'nomadic':village?'village':type==='pass'?'border-pass':type==='fortress'?'fortress'
      :['port','island-port'].includes(type)?'seaport':type==='lake-port'?'lake-port':recipe.harbor?.kind==='river'?'river-port':type==='river-city'?'river-town'
      :type==='oasis'?'oasis':['rolling','steep','escarpment'].includes(recipe.terrain?.slopeClass)?'hill-town':'market-town';
    const label={nomadic:'游牧营地',village:'村落','border-pass':'边关',fortress:'要塞',seaport:'海港城市',
      'lake-port':'湖港城市','river-port':'内河港口城市','river-town':'河岸城市',oasis:'绿洲城市','hill-town':'山地城市','market-town':'商贸城市'}[kind];
    const defensive=['ancient','medieval','early-modern','preindustrial'].includes(recipe.era)&&!small;
    const capital=culture.isCapital||(recipe.urban?.coreZones||[]).some(c=>c.role==='government-core');
    const layers=defensive?(capital||population>15000||['fortress','border-pass'].includes(kind)?2:1):0;
    const barbicans=defensive&&recipe.era!=='ancient'&&(capital||population>8500||['fortress','border-pass'].includes(kind));
    const tradition=traditions[style];
    const plan=small?mobile?'camp':'village':['industrial','contemporary'].includes(recipe.era)?'orthogonal':tradition.plan;
    const material=materialProfile(recipe,style);
    return {...styles[style],roof:material.roof,material,palace:palaceDesign(recipe,style),tradition,plan,capital,style,kind,label:small&&kind==='seaport'?'港口聚落':label,mobile,village,small,defence:{layers,barbicans},
      prefix:Array.from(recipe.name||'当地').slice(0,3).join(''),
      buildingCount:Math.max(mobile?10:18,Math.min(6500,Math.round(population/(mobile?10:recipe.era==='contemporary'?26:11))))};
  }
  const rect=(x,y,w,h)=>[[x-w/2,y-h/2],[x+w/2,y-h/2],[x+w/2,y+h/2],[x-w/2,y+h/2]];
  const circle=(x,y,r)=>Array.from({length:16},(_,i)=>[x+Math.cos(i*Math.PI/8)*r,y+Math.sin(i*Math.PI/8)*r]);
  function template(kind,style,population=0,design={family:'axial-courts',variant:0}) {
    // Each compound fits in [-1,1]². Its yard is reserved before ordinary lots.
    let parts=[];
    const add=(points,roof='hip')=>parts.push({points,roof});
    if(kind==='royal-palace'||kind==='imperial-palace'){
      const {family,variant}=design,shift=(variant-1)*.09,large=population>=100000;
      if(family==='axial-courts'){
        const halls=variant===0?[-.68,-.12,.44]:variant===1?[-.61,.10]:[-.70,-.30,.29];
        halls.forEach((y,i)=>add(rect(shift,y,i===1?1.10:.90,.22)));
        for(const x of [-.65,.65])for(const y of [-.46,.10,.56])add(rect(x,y,.17,.35));
        add(rect(shift,.80,.45,.16),'gate');
        if(variant===1){add(rect(-.30,-.66,.27,.19));add(rect(.30,-.66,.27,.19));}
        if(large)for(const x of [-1.04,1.04])for(const y of [-.68,-.25,.23,.68])add(rect(x,y,.31,.25));
      }else if(family==='merchant-palazzo'){
        add(rect(shift,-.58,1.70,.40),'hip');
        for(const x of [-.81,.81])add(rect(x,.02,.25,.79),'hip');
        add(rect(-.44,.52,.66,.22),'gable');add(rect(.46,.52,.65,.22),'gable');
        add(rect(0,.82,.48,.16),'gate');
        for(const x of [-.66,-.22,.22,.66])add(rect(x,-.87,.15,.18),'tower');
        if(variant===1)add(rect(0,.08,.60,.18),'flat');
        if(large)for(const x of [-1.09,1.09])for(const y of [-.64,-.22,.20,.62])add(rect(x,y,.24,.27));
      }else if(family==='peristyle'){
        add(rect(shift,-.63,1.65,.32),'gable');
        for(const x of [-.83,.83])add(rect(x,-.03,.20,.88),'gable');
        for(const x of [-.52,.52])add(rect(x,.43,.45,.18),'flat');
        add(rect(shift,.82,.48,.18),'gate');
        if(variant===1)add(rect(0,.23,.68,.19),'flat');
        if(variant===2)add(circle(-.46,-.62,.18),'dome');
        if(large)for(const x of [-1.10,1.10])for(const y of [-.70,-.30,.12,.56])add(rect(x,y,.22,.27));
      }else if(family==='hall-and-bailey'){
        add(rect(-.15,-.42,1.24,.39),'gable');add(rect(.59,-.05,.23,.70),'gable');
        add(rect(-.68,.31,.29,.60),'gable');add(rect(.18,.45,.52,.24),'gable');
        for(const x of [-.72,.63])add(rect(x,-.68,.22,.22),'tower');
        add(rect(0,.80,.39,.21),'gate');add(rect(-.34,.10,.25,.22),'hip');
        if(variant===2)add(rect(.50,.56,.30,.22),'gable');
        if(large)for(const x of [-1.08,1.03])for(const y of [-.66,-.22,.20,.63])add(rect(x,y,.25,.27),'gable');
      }else if(family==='castle-palace'){
        add(rect(-.14,-.42,1.25,.36),'gable');
        add(rect(-.68,.08,.28,.64),'gable');add(rect(.53,-.04,.23,.53),'gable');
        for(const [x,y] of [[-.73,-.61],[.61,-.58],[-.77,.62],[.67,.62]])add(variant===1?rect(x,y,.21,.21):circle(x,y,.14),'tower');
        add(rect(-.18,.81,.38,.20),'gate');
        add(rect(.30,.46,.49,.22));
        if(large)for(const x of [-1.08,1.03])for(const y of [-.66,-.22,.20,.63])add(rect(x,y,.25,.27),'gable');
      }else if(family==='terraced-court'){
        add(rect(shift,-.36,1.55,.28));
        for(const x of [-.81,-.30,.27,.80])add(rect(x,.36,.28,.28));
        add(rect(-.98,-.23,.20,.58),'tower');add(rect(.98,-.21,.20,.62),'tower');
        add(rect(shift,.81,.42,.18),'gate');
        if(large)for(const x of [-1.09,-.55,0,.55,1.09]){add(rect(x,-.77,.28,.20));add(rect(x,.65,.30,.16));}
      }else{
        add(rect(shift,-.54,1.05,.35));
        for(const [x,y] of [[-.75,-.18],[.73,-.05],[-.68,.48],[.64,.52]])add(rect(x,y,.37,.26));
        add(rect(-.29,.12,.36,.22));add(rect(.23,.20,.33,.20));add(rect(0,.83,.38,.16),'gate');
        if(large)for(const x of [-1.1,1.08])for(const y of [-.67,-.19,.34,.73])add(rect(x,y,.23,.21));
      }
    }else if(kind==='council-house'){
      add([[-.65,-.50],[.40,-.50],[.72,-.18],[.72,.27],[.40,.57],[-.65,.57]],'dome');
      add(rect(0,-.68,1.34,.20),'flat');
    }else if(kind==='holy-tomb'){
      add(circle(0,-.15,.43),'dome');add(rect(-.59,.10,.24,.90));add(rect(.59,.10,.24,.90));add(rect(0,.64,1.35,.22));
    }else if(kind==='observatory-sanctuary'){
      add(circle(0,-.13,.53),'tower');add(rect(0,.62,1.3,.24));for(const x of [-.64,.64])add(rect(x,-.1,.20,.95));
    }else if(['great-sanctuary','pilgrimage-monastery','water-sanctuary','harbor-shrine'].includes(kind)){
      add(rect(0,-.53,1.35,.32));for(const x of [-.60,.60])add(rect(x,0,.23,1.06));
      add(rect(-.38,.62,.54,.20));add(rect(.38,.62,.54,.20));
      if(kind==='great-sanctuary')add(circle(0,-.52,.26),'dome');
      if(kind==='pilgrimage-monastery')for(const x of [-.38,.38])add(rect(x,-.12,.24,.20));
      if(kind==='harbor-shrine')add(circle(.55,-.50,.16),'tower');
    }else if(kind==='sanctuary') {
      if(styles[style]?.temple==='colonnade') {
        add(rect(0,-.12,.82,1.03));
        for(const x of [-.55,.55])for(let i=0;i<5;i++)add(rect(x,-.56+i*.22,.09,.1),'flat');
        add(rect(0,.58,1.28,.16),'flat');
      } else if(styles[style]?.temple==='longhouse') {
        add([[-.25,-.72],[.25,-.72],[.25,.12],[.52,.12],[.52,.36],[.25,.36],[.25,.68],[-.25,.68],[-.25,.36],[-.52,.36],[-.52,.12],[-.25,.12]]);
        add(rect(0,-.66,.31,.31),'tower');
      } else if(style==='nomadic')add(circle(0,0,.46),'dome');
      else {add(rect(0,-.46,1.26,.3));add(rect(-.48,.05,.27,.8));add(rect(.48,.05,.27,.8));add(rect(0,.48,1.22,.24));add(circle(0,-.44,.17),'dome');}
    } else if(kind==='caravanserai') {
      add(rect(0,-.52,1.5,.28),'flat');add(rect(-.6,0,.28,1.06),'flat');add(rect(.6,0,.28,1.06),'flat');
      add(rect(-.4,.52,.48,.28),'flat');add(rect(.4,.52,.48,.28),'flat');
    } else if(kind==='barracks') {
      for(const y of [-.55,0,.55])add(rect(0,y,1.36,.24));
    } else if(kind==='granary'||kind==='customs-house') {
      add(rect(-.3,0,.48,1.2));add(rect(.34,0,.48,1.2));add(rect(0,.57,.9,.15),'flat');
    } else if(kind==='lord-manor'){
      add(rect(0,-.4,1.35,.36));add(rect(-.48,.08,.26,.68));add(rect(.48,.08,.26,.68));
      add(rect(.45,.55,.42,.22));add(rect(-.48,-.43,.30,.30),'tower');
    } else if(kind==='assembly-tent')add(circle(0,0,.52),'dome');
    else if(kind==='town-hall') {add(rect(0,-.25,1.45,.39));add(rect(-.53,.12,.25,.42));add(rect(.53,.12,.25,.42));add(rect(0,-.25,.23,.23),'tower');}
    else {add(rect(0,-.28,1.5,.36));for(let i=0;i<6;i++)add(rect((i-2.5)*.23,.4,.16,.19),'flat');}
    return parts;
  }
  window.WorldAtlasCityCharacter=Object.freeze({profile,template});
})();
