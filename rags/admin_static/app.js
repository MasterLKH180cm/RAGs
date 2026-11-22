/* Self-hosted admin client. Never stores bearer credentials in browser storage. */
(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const titles = {overview:'總覽',documents:'知識文件',explorer:'檢索探索',agents:'自主代理',observability:'可觀測性',settings:'設定與權限'};
  const state = {csrf:'',tenant:'',permissions:[],view:'overview',offset:0,search:'',limit:20,deleteTarget:null,agentRun:null,loading:false,authenticated:false,epoch:0};
  const pending=new Set();
  let searchTimer,toastTimer;
  function el(tag,klass,label){
    const item=document.createElement(tag); if(klass)item.className=klass;
    if(label!==undefined&&label!==null)item.textContent=String(label);return item;
  }
  function txt(id,value){$(id).textContent=value===null||value===undefined?'—':String(value);}
  function n(value){return Number.isFinite(value)?new Intl.NumberFormat('zh-TW').format(value):'—';}
  function date(value){const d=new Date(value);return Number.isFinite(d.getTime())
    ?new Intl.DateTimeFormat('zh-TW',{dateStyle:'medium',timeStyle:'short'}).format(d):'—';}
  function ms(value){return Number.isFinite(value)?value.toFixed(1)+' ms':'—';}
  function notice(message,isError=false){
    if(!message)return;
    const node=$('#global-notice');node.textContent=message;node.classList.toggle('error',isError);
    node.classList.add('visible');clearTimeout(toastTimer);
    toastTimer=setTimeout(()=>node.classList.remove('visible'),4200);
  }
  function resetTenantUI(){
    // Clear ALL tenant content before revealing a different identity. Abort
    // requests in flight and ensure late responses cannot repopulate the DOM.
    state.epoch+=1;
    for(const controller of pending)controller.abort();
    pending.clear();
    state.csrf='';state.tenant='';state.permissions=[];state.offset=0;
    state.search='';state.loading=false;state.deleteTarget=null;state.agentRun=null;
    clearTimeout(searchTimer);
    for(const id of ['#query-citations','#documents-body',
        '#activity-list','#feature-list','#permission-list','#overview-chart','#monitor-chart']){
      $(id).replaceChildren();
    }
    $('#query-output').hidden=true;$('#query-empty').hidden=false;
    $('#agent-output').hidden=true;$('#agent-empty').hidden=false;
    $('#agent-approval').hidden=true;
    $('#login-error').hidden=true;$('#login-error').textContent='';
    $('#query-form').reset();$('#add-form').reset();$('#delete-form').reset();$('#agent-form').reset();
    $('#document-search').value='';$('#delete-confirm').value='';
    $('#delete-submit').disabled=true;
    // Async request finalizers skip a changed identity by design; restore all
    // tenant-specific controls here instead of leaving them busy for the next login.
    $('#query-submit').disabled=false;$('#query-submit').textContent='開始檢索 ↗';
    $('#add-submit').disabled=false;$('#add-submit').textContent='建立並索引';
    $('#agent-submit').disabled=false;$('#agent-submit').textContent='執行代理任務 ↗';
    $('#agent-approve').disabled=false;$('#agent-deny').disabled=false;
    for(const id of ['#add-dialog','#delete-dialog'])if($(id).open)$(id).close();
    for(const id of ['#query-answer','#citation-count','#query-status',
        '#document-count','#document-range','#stat-documents','#stat-chunks',
        '#stat-revisions','#stat-images','#setting-tenant','#setting-auth',
        '#setting-storage','#setting-embedding','#setting-generation',
        '#tenant-sidebar','#session-expiry','#agent-answer','#agent-run-id',
        '#agent-steps','#agent-tool-count','#agent-stop-reason','#agent-pending-tool',
        '#agent-pending-args','#agent-status','#agent-availability','#monitor-requests',
        '#monitor-errors','#monitor-p50','#monitor-p95','#exported-spans',
        '#dropped-spans','#failed-exports','#collector-status'])txt(id,'—');
    $('#global-notice').classList.remove('visible');
  }
  function showLogin(){
    resetTenantUI();state.authenticated=false;
    $('#workspace').hidden=true;$('#auth-screen').hidden=false;$('#admin-token').value='';
  }
  function showApp(info){
    resetTenantUI();state.authenticated=true;
    state.csrf=info.csrf;state.tenant=info.tenant;state.permissions=info.permissions||[];
    txt('#tenant-sidebar',info.tenant);txt('#setting-tenant',info.tenant);
    txt('#session-expiry','有效至 '+date(info.expires_at));
    $('#auth-screen').hidden=true;$('#workspace').hidden=false;changeView('overview');
  }
  class APIError extends Error{constructor(status,code){super(code||'request_failed');this.status=status;}}
  class StaleSession extends Error{constructor(){super('stale_session');}}
  async function api(path,{method='GET',body,headers={}}={}){
    const epoch=state.epoch,controller=new AbortController();
    pending.add(controller);
    const h={Accept:'application/json',...headers};if(method!=='GET')h['X-CSRF-Token']=state.csrf;
    if(body!==undefined){body=JSON.stringify(body);h['Content-Type']='application/json';}
    try{
      const response=await fetch('/admin/api/'+path,{method,headers:h,body,credentials:'same-origin',
        cache:'no-store',signal:controller.signal});
      if(epoch!==state.epoch)throw new StaleSession();
      if(!response.ok){
        let code='request_failed';
        try{code=(await response.json()).detail?.code||code;}catch(_){}
        if(epoch!==state.epoch)throw new StaleSession();
        if(response.status===401)showLogin();
        throw new APIError(response.status,code);
      }
      if(response.status===204)return null;
      const data=await response.json();
      if(epoch!==state.epoch)throw new StaleSession();
      return data;
    }catch(error){
      if(epoch!==state.epoch||error.name==='AbortError')throw new StaleSession();
      if(error instanceof APIError||error instanceof StaleSession)throw error;
      throw new Error('network_error');
    }finally{pending.delete(controller);}
  }
  function errorText(error){
    if(error instanceof StaleSession)return '';
    if(!(error instanceof APIError))return '連線暫時中斷，請稍後再試。';
    if(error.status===401)return '工作階段已到期，請重新登入。';
    if(error.status===403)return '權限不足或操作驗證失敗。';
    if(error.status===409)return '文件已更新，請重新整理後重試。';
    if(error.status===422)return '輸入資料有誤，請檢查格式。';
    if(error.status===503)return '後端或身分驗證服務暫時不可用。';
    return '操作未完成，請稍後重試。';
  }
  function busy(button,enabled,label){
    button.disabled=enabled;if(label!==undefined)button.textContent=label;
  }
  function closeMenu(){
    $('#sidebar').classList.remove('open');$('#nav-scrim').hidden=true;
    $('#menu-toggle').setAttribute('aria-expanded','false');
  }
  function changeView(view){
    if(!(view in titles))return;state.view=view;
    $$('.page-view').forEach(p=>p.hidden=p.id!=='view-'+view);
    $$('.nav-item').forEach(b=>{
      const active=b.dataset.view===view;b.classList.toggle('selected',active);
      if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');
    });
    txt('#breadcrumb',titles[view]);document.title=titles[view]+' · RAGs Control Center';
    closeMenu();$('#main-content').focus({preventScroll:true});
    window.scrollTo({top:0,behavior:'auto'});
    if(view==='overview'||view==='observability')refreshOverview();
    if(view==='documents')loadDocuments();
    if(view==='settings')loadOperations();
    if(view==='agents')loadAgentCapabilities();
  }
  function drawChart(target,series){
    const root=$(target);root.replaceChildren();
    if(!Array.isArray(series)||!series.length){
      root.append(el('p','empty-state','目前沒有流量資料。'));return;
    }
    const peak=Math.max(1,...series.map(d=>Number(d.requests)||0));
    series.forEach((point,index)=>{
      const count=Number(point.requests)||0,errors=Number(point.errors)||0;
      const g=el('div','chart-group');g.title=date(point.at)+' · '+count+' 次請求 / '+errors+' 次錯誤';
      g.setAttribute('aria-label',g.title);
      const bar=el('div','chart-column');bar.style.height=Math.max(2,count/peak*100)+'%';g.append(bar);
      if(errors){const err=el('div','chart-column chart-error');err.style.height=Math.max(2,errors/peak*100)+'%';g.append(err);}
      if(index===0||index===5||index===series.length-1){
        const d=new Date(point.at),valid=Number.isFinite(d.getTime());
        const label=valid?new Intl.DateTimeFormat('zh-TW',{hour:'2-digit',minute:'2-digit'}).format(d):'';
        g.append(el('span','chart-time',label));
      }
      root.append(g);
    });
  }
  function drawActivity(events){
    const root=$('#activity-list');root.replaceChildren();
    if(!events||!events.length){root.append(el('li','empty-state','沒有近期活動。新增文件後會顯示在這裡。'));return;}
    events.forEach(row=>{
      const li=el('li'),icon=el('span','activity-symbol',row.action==='delete'?'×':'↗');
      icon.setAttribute('aria-hidden','true');
      const copy=el('div','activity-copy');
      copy.append(el('strong','',(row.action==='delete'?'已刪除':'已更新')+' · '+(row.document_id||'來源文件')));
      copy.append(el('span','',date(row.timestamp)));li.append(icon,copy);root.append(li);
    });
  }
  function renderOverview(info){
    const c=info.counts||{},m=info.telemetry||{},otel=m.otel||{};
    txt('#stat-documents',n(c.documents));txt('#stat-chunks',n(c.chunks));
    txt('#stat-revisions',n(c.revisions));txt('#stat-images',n(c.images));
    drawActivity(info.activity);drawChart('#overview-chart',m.series);
    drawChart('#monitor-chart',m.series);txt('#monitor-requests',n(m.requests_5m));
    txt('#monitor-errors',Number.isFinite(m.error_rate_5m)?m.error_rate_5m.toFixed(1)+'%':'—');
    txt('#monitor-p50',ms(m.p50_ms_5m));txt('#monitor-p95',ms(m.p95_ms_5m));
    txt('#exported-spans',n(otel.exported));txt('#dropped-spans',n(otel.dropped));
    txt('#failed-exports',n(otel.export_failures));
    const chip=$('#collector-status');chip.textContent=otel.enabled?'OTLP 已啟用':'OTLP 未啟用';
    chip.classList.toggle('online',!!otel.enabled&&!otel.export_failures);
  }
  async function refreshOverview(){
    if(!state.authenticated||state.loading)return;
    const epoch=state.epoch;state.loading=true;
    try{const data=await api('overview');if(epoch===state.epoch)renderOverview(data);}
    catch(e){if(epoch===state.epoch)notice(errorText(e),true);}
    finally{if(epoch===state.epoch)state.loading=false;}
  }
  function renderDocuments(data){
    const root=$('#documents-body');root.replaceChildren();
    txt('#document-count','共 '+n(data.total)+' 份文件');
    if(!data.items.length){
      const tr=el('tr'),td=el('td','empty-state',state.search?
        '沒有符合搜尋條件的文件。':'目前沒有任何文件，可以新增第一份來源。');
      td.colSpan=4;tr.append(td);root.append(tr);
    }
    data.items.forEach(doc=>{
      const tr=el('tr'),name=el('td');
      name.append(el('strong','doc-title',doc.title),el('span','doc-id',doc.id));
      const source=el('td');source.append(el('span','source-tag',doc.source));
      const updated=el('td','',date(doc.updated_at));
      const actions=el('td','right'),remove=el('button','row-button danger','刪除');
      remove.type='button';remove.setAttribute('aria-label','刪除 '+doc.title);
      remove.addEventListener('click',()=>openDelete(doc));actions.append(remove);
      tr.append(name,source,updated,actions);root.append(tr);
    });
    const start=data.total?data.offset+1:0,end=Math.min(data.total,data.offset+data.items.length);
    txt('#document-range','顯示 '+start+'–'+end+' / '+n(data.total));
    $('#page-prev').disabled=state.offset===0;
    $('#page-next').disabled=state.offset+state.limit>=data.total;
  }
  async function loadDocuments(){
    const epoch=state.epoch,root=$('#documents-body');root.replaceChildren();
    const tr=el('tr'),td=el('td','empty-state','正在讀取文件…');td.colSpan=4;tr.append(td);root.append(tr);
    const qs=new URLSearchParams({search:state.search,offset:String(state.offset),limit:String(state.limit)});
    try{const data=await api('documents?'+qs.toString());if(epoch===state.epoch)renderDocuments(data);}
    catch(e){if(epoch===state.epoch){td.textContent=errorText(e);root.replaceChildren(tr);}}
  }
  function openDelete(doc){
    state.deleteTarget=doc;txt('#delete-target',doc.id);$('#delete-confirm').value='';
    $('#delete-submit').disabled=true;$('#delete-dialog').showModal();$('#delete-confirm').focus();
  }
  async function loadOperations(){
    const epoch=state.epoch;
    try{
      const d=await api('operations');
      if(epoch!==state.epoch)return;
      txt('#setting-auth',d.auth_mode);txt('#setting-storage',d.storage);
      txt('#setting-embedding',d.embedding_provider);
      txt('#setting-generation',d.generation_provider);
      const perms=$('#permission-list');perms.replaceChildren();
      (d.permissions||[]).forEach(p=>perms.append(el('span','scope-chip',p)));
      const flags=$('#feature-list');flags.replaceChildren();
      Object.entries(d.features||{}).sort((a,b)=>a[0].localeCompare(b[0])).forEach(pair=>{
        const item=el('div','feature-item');
        item.append(el('span','',pair[0]),el('span',pair[1]?'feature-state':'feature-state disabled',pair[1]?'已啟用':'停用'));
        flags.append(item);
      });
    }catch(e){if(epoch===state.epoch)notice(errorText(e),true);}
  }
  function renderQuery(result){
    $('#query-empty').hidden=true;$('#query-output').hidden=false;
    const answer=result.answer||(result.outcome==='insufficient_evidence'
      ?'目前沒有足夠的授權證據回答此問題。'
      :'已完成檢索；未開啟模型回答，請查閱來源證據。');
    txt('#query-answer',answer);
    txt('#query-status',result.outcome==='insufficient_evidence'?'證據不足':'完成');
    const cites=Array.isArray(result.citations)?result.citations:[];
    txt('#citation-count',cites.length+' 份證據');
    const root=$('#query-citations');root.replaceChildren();
    if(!cites.length)root.append(el('p','empty-state','沒有可引用的來源。'));
    cites.forEach(c=>{
      const card=el('article','citation-card'),head=el('div','citation-title');
      head.append(el('span','',c.title||c.document_id||'來源文件'));
      if(Number.isFinite(c.score))head.append(el('span','citation-score','Score '+c.score.toFixed(3)));
      card.append(head,el('p','citation-meta',[c.source,c.section].filter(Boolean).join(' · ')));
      // The answer references exact [chunk_id] values. Expose the same ID on
      // each evidence card so citations remain auditable even with duplicate titles.
      if(typeof c.chunk_id==='string'&&c.chunk_id){
        const reference=el('div','citation-reference');
        reference.append(el('span','citation-reference-label','引用 ID'),
          el('code','citation-id','['+c.chunk_id+']'));
        card.append(reference);
      }
      card.append(el('div','citation-text',c.text||'沒有可顯示的摘錄。'));root.append(card);
    });
  }
  async function signIn(event){
    event.preventDefault();const input=$('#admin-token'),token=input.value.trim();input.value='';
    if(!token){txt('#login-error','請輸入具有管理員權限的權杖。');$('#login-error').hidden=false;return;}
    const submit=$('#login-submit');busy(submit,true,'驗證中…');$('#login-error').hidden=true;
    try{
      const res=await fetch('/admin/api/session',{method:'POST',headers:{Authorization:'Bearer '+token},
        credentials:'same-origin',cache:'no-store'});
      if(!res.ok)throw new APIError(res.status);
      showApp(await api('session'));notice('已安全登入。');
    }catch(e){
      txt('#login-error',e.status===403?'需要管理員權限與安全連線。':
        '登入失敗，請確認權杖與身分驗證服務。');$('#login-error').hidden=false;
    }finally{busy(submit,false,'安全登入 →');}
  }
  async function logout(){
    try{await api('session',{method:'DELETE'});}catch(_){}
    showLogin();notice('已登出工作區。');
  }
  async function addSource(event){
    event.preventDefault();const title=$('#add-title').value.trim(),content=$('#add-content').value.trim();
    if(!title||!content){notice('請填寫文件名稱及內容。',true);return;}
    const body={title,text:content,source:$('#add-source').value.trim()||'dashboard'};
    if($('#add-id').value.trim())body.document_id=$('#add-id').value.trim();
    const epoch=state.epoch,btn=$('#add-submit');busy(btn,true,'正在索引…');
    try{
      await api('documents',{method:'POST',body});
      if(epoch!==state.epoch)return;$('#add-dialog').close();$('#add-form').reset();
      notice('文件與索引已建立。');state.offset=0;await Promise.all([loadDocuments(),refreshOverview()]);
    }catch(e){if(epoch===state.epoch)notice(errorText(e),true);}
    finally{if(epoch===state.epoch)busy(btn,false,'建立並索引');}
  }
  async function removeSource(event){
    event.preventDefault();const doc=state.deleteTarget;
    if(!doc||$('#delete-confirm').value.trim()!==doc.id)return;
    const epoch=state.epoch,btn=$('#delete-submit');busy(btn,true,'刪除中…');
    try{
      await api('documents/'+encodeURIComponent(doc.id),{method:'DELETE',headers:{'If-Match':doc.revision}});
      if(epoch!==state.epoch)return;
      state.deleteTarget=null;$('#delete-dialog').close();notice('文件已刪除。');
      await Promise.all([loadDocuments(),refreshOverview()]);
    }catch(e){if(epoch===state.epoch)notice(errorText(e),true);}
    finally{if(epoch===state.epoch)busy(btn,false,'確認刪除');}
  }
  async function runQuery(event){
    event.preventDefault();const question=$('#query-question').value.trim();
    if(question.length<2){notice('問題至少需要兩個字元。',true);return;}
    const epoch=state.epoch,btn=$('#query-submit');busy(btn,true,'正在檢索…');txt('#query-status','執行中');
    try{
      const result=await api('query',{method:'POST',body:{question,mode:$('#query-mode').value,
        top_k:Number($('#query-topk').value),generate:$('#query-generate').checked}});
      if(epoch!==state.epoch)return;
      renderQuery(result);
      notice('檢索完成。');
    }catch(e){if(epoch===state.epoch){txt('#query-status','執行失敗');notice(errorText(e),true);}}
    finally{if(epoch===state.epoch)busy(btn,false,'開始檢索 ↗');}
  }
  async function loadAgentCapabilities(){
    const epoch=state.epoch, select=$('#agent-model');select.replaceChildren();
    const placeholder=el('option','', '檢查可用模型…');placeholder.value='';select.append(placeholder);
    try{
      const data=await api('agents/capabilities');
      if(epoch!==state.epoch)return;
      select.replaceChildren();
      const models=Array.isArray(data.models)?data.models:[];
      for(const model of models){
        const option=el('option','',model);option.value=model;select.append(option);
      }
      if(!models.length){
        const empty=el('option','', '尚未設定模型');empty.value='';select.append(empty);
      }
      $('#agent-submit').disabled=!models.length;
      txt('#agent-availability',models.length?('已授權 · '+models.length+' 個模型'):'沒有可用的模型');
    }catch(e){
      if(epoch!==state.epoch)return;
      $('#agent-submit').disabled=true;txt('#agent-availability','代理功能未啟用');
      const empty=el('option','', '代理服務尚不可用');empty.value='';select.replaceChildren(empty);
    }
  }
  function renderAgent(result){
    state.agentRun=result;$('#agent-empty').hidden=true;$('#agent-output').hidden=false;
    txt('#agent-status',result.status||'未知');
    txt('#agent-run-id',result.run_id);txt('#agent-steps',n(result.steps));
    txt('#agent-tool-count',n(result.tool_calls));
    txt('#agent-stop-reason',result.stop_reason||'—');
    txt('#agent-answer',result.answer||(
      result.status==='waiting_for_approval'?'代理已暫停，等待人工核准。'
        :result.status==='stopped'?'代理達到停止條件。':'代理尚未提供最終回答。'));
    const gate=$('#agent-approval');
    const pending=result.pending_approval;
    gate.hidden=!(result.status==='waiting_for_approval'&&pending);
    if(!gate.hidden){
      txt('#agent-pending-tool',pending.name);
      txt('#agent-pending-args',JSON.stringify(pending.arguments,null,2));
    }else{
      txt('#agent-pending-tool','—');txt('#agent-pending-args','');
    }
  }
  async function startAgent(event){
    event.preventDefault();const epoch=state.epoch;
    const goal=$('#agent-goal').value.trim(),model=$('#agent-model').value;
    const session_id=$('#agent-session').value.trim();
    if(goal.length<2||!model||!session_id){
      notice('請填寫目標、模型與有效的工作階段 ID。',true);return;
    }
    const button=$('#agent-submit');busy(button,true,'代理執行中…');
    try{
      const result=await api('agents/runs',{method:'POST',body:{goal,model,session_id}});
      if(epoch!==state.epoch)return;
      renderAgent(result);notice(result.status==='waiting_for_approval'
        ?'任務已暫停，請確認受保護的工具操作。':'已更新代理任務狀態。');
    }catch(e){if(epoch===state.epoch)notice(errorText(e),true);}
    finally{if(epoch===state.epoch)busy(button,false,'執行代理任務 ↗');}
  }
  async function decideAgent(approved){
    const run=state.agentRun,epoch=state.epoch;
    if(!run||run.status!=='waiting_for_approval')return;
    const approve=$('#agent-approve'),deny=$('#agent-deny');
    approve.disabled=true;deny.disabled=true;
    try{
      const result=await api('agents/runs/'+encodeURIComponent(run.run_id)+'/resume',{
        method:'POST',body:{expected_revision:run.revision,approved},
      });
      if(epoch!==state.epoch)return;
      renderAgent(result);notice(approved?'已核准，代理繼續執行。':'已拒絕受保護操作。');
    }catch(e){if(epoch===state.epoch)notice(errorText(e),true);}
    finally{if(epoch===state.epoch){approve.disabled=false;deny.disabled=false;}}
  }
  async function bootstrap(){
    const epoch=state.epoch;
    try{const info=await api('session');if(epoch===state.epoch)showApp(info);}
    catch(_){if(epoch===state.epoch)showLogin();}
  }
  function init(){
    try{const p=localStorage.getItem('rags-admin-theme');
      if(p==='dark'||p==='light')document.body.dataset.theme=p;}catch(_){}
    $('#theme-toggle').addEventListener('click',()=>{
      const mode=document.body.dataset.theme==='dark'?'light':'dark';
      document.body.dataset.theme=mode;
      try{localStorage.setItem('rags-admin-theme',mode);}catch(_){}
    });
    $('#login-form').addEventListener('submit',signIn);
    $('#logout-button').addEventListener('click',logout);
    $('#show-token').addEventListener('click',()=>{
      const input=$('#admin-token'),shown=input.type==='password';
      input.type=shown?'text':'password';$('#show-token').textContent=shown?'隱藏':'顯示';
      $('#show-token').setAttribute('aria-pressed',String(shown));
    });
    $$('.nav-item').forEach(b=>b.addEventListener('click',()=>changeView(b.dataset.view)));
    $$('[data-go]').forEach(b=>b.addEventListener('click',()=>changeView(b.dataset.go)));
    $('#refresh-button').addEventListener('click',()=>{
      if(state.view==='documents')loadDocuments();
      else if(state.view==='settings')loadOperations();else refreshOverview();
    });
    $('#menu-toggle').addEventListener('click',()=>{
      const open=!$('#sidebar').classList.contains('open');
      $('#sidebar').classList.toggle('open',open);$('#nav-scrim').hidden=!open;
      $('#menu-toggle').setAttribute('aria-expanded',String(open));
    });
    $('#mobile-close').addEventListener('click',closeMenu);
    $('#nav-scrim').addEventListener('click',closeMenu);
    document.addEventListener('keydown',e=>{if(e.key==='Escape')closeMenu();});
    $$('[data-open-add]').forEach(b=>b.addEventListener('click',()=>{
      $('#add-dialog').showModal();$('#add-title').focus();
    }));
    $$('[data-close-dialog]').forEach(b=>b.addEventListener('click',()=>{
      $('#'+b.dataset.closeDialog).close();
    }));
    $('#add-form').addEventListener('submit',addSource);
    $('#delete-form').addEventListener('submit',removeSource);
    $('#query-form').addEventListener('submit',runQuery);
    $('#agent-form').addEventListener('submit',startAgent);
    $('#agent-approve').addEventListener('click',()=>decideAgent(true));
    $('#agent-deny').addEventListener('click',()=>decideAgent(false));
    $('#delete-confirm').addEventListener('input',()=>{
      $('#delete-submit').disabled=!state.deleteTarget||
        $('#delete-confirm').value.trim()!==state.deleteTarget.id;
    });
    $('#document-search').addEventListener('input',e=>{
      clearTimeout(searchTimer);const value=e.target.value.trim();
      searchTimer=setTimeout(()=>{state.search=value;state.offset=0;loadDocuments();},280);
    });
    $('#page-prev').addEventListener('click',()=>{
      state.offset=Math.max(0,state.offset-state.limit);loadDocuments();
    });
    $('#page-next').addEventListener('click',()=>{
      state.offset+=state.limit;loadDocuments();
    });
    setInterval(()=>{
      if(state.authenticated&&(state.view==='overview'||state.view==='observability')
        &&!document.hidden)refreshOverview();
    },30000);
    bootstrap();
  }
  init();
})();
