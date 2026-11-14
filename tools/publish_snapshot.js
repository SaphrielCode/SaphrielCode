// Run through functions.exec after reviewing this file; tools are provided by that runtime.
// Large file data stays inside orchestration, never printed to the conversation.
const ROOT='C:/Users/leeme/SaphrielComDevice';
function data(r){if(r.isError)throw Error(r.content?.filter(x=>x.type==='text').map(x=>x.text.slice(0,700)).join('\n'));const s=r.structuredContent;if(s?.content)return JSON.parse(s.content);if(s)return s;throw Error('Missing structured GitHub response');}
function b64ascii(value){const s=JSON.stringify(value).replace(/[\u007f-\uffff]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0')),abc='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';let out='';for(let i=0;i<s.length;i+=3){const a=s.charCodeAt(i),b=s.charCodeAt(i+1),c=s.charCodeAt(i+2),n=(a<<16)|((b||0)<<8)|(c||0);out+=abc[(n>>>18)&63]+abc[(n>>>12)&63]+(i+1<s.length?abc[(n>>>6)&63]:'=')+(i+2<s.length?abc[n&63]:'=');}return out;}
async function persist(name,value){if(!/^[A-Za-z0-9_]+\.json$/.test(name))throw Error('Invalid receipt filename');const encoded=b64ascii(value);const cmd="@'\nimport base64\nfrom pathlib import Path\nPath('results/"+name+"').write_bytes(base64.b64decode('"+encoded+"'))\n'@ | .venv-data\\Scripts\\python.exe -";const r=await tools.exec_command({cmd,workdir:ROOT,sandbox_permissions:'require_escalated',max_output_tokens:100});if(r.exit_code!==0)throw Error('Could not persist publication receipt');}
async function readJson(path){const r=await tools.mcp__node_repl__js({code:'nodeRepl.write(await (await import("node:fs/promises")).readFile('+JSON.stringify(path)+',"utf8"));',title:'Reading publication manifest'});if(r.isError)throw Error('Cannot read publication manifest');return JSON.parse(r.content.find(x=>x.type==='text').text);}
const plan=await readJson(ROOT+'/results/GitPublicationPlan.json');
if(plan.root.replaceAll('\\','/').toLowerCase()!==ROOT.toLowerCase())throw Error('Unexpected publication root');
const repo=plan.repository;if(repo!=='SaphrielCode/SaphrielCode')throw Error('Unexpected destination');
const api='https://api.github.com/repos/'+repo;
const ref=data(await tools.mcp__codex_apps__github_fetch({url:api+'/git/ref/heads/main'}));
if(ref.object.sha!==plan.expected_remote_base)throw Error('Remote changed. Reconcile before publishing; no force update allowed.');
const commit=data(await tools.mcp__codex_apps__github_fetch({url:api+'/git/commits/'+ref.object.sha}));
const oldTree=data(await tools.mcp__codex_apps__github_fetch({url:api+'/git/trees/'+commit.tree.sha+'?recursive=1'}));
if(oldTree.truncated)throw Error('Cannot verify full existing tree');
const existing=new Set(oldTree.tree.filter(e=>e.type==='blob').map(e=>e.sha));let cache={done:[]};
try{cache=await readJson(ROOT+'/results/GitBlobCache.json');}catch{}
for(const sha of cache.done||[])existing.add(sha);
let completed=0,uploaded=0,bytes=0,lastNotice=Date.now();
for(const item of plan.blobs){
 if(!/^[a-f0-9]{40}$/.test(item.sha)||item.bytes>4*1024*1024||item.bytes<0||/^(?:[A-Za-z]:|[/\\])|(?:^|[/\\])\.\.(?:[/\\]|$)/.test(item.local_file))throw Error('Invalid artifact chunk metadata');
 if(!existing.has(item.sha)){
  const full=ROOT+'/'+item.local_file;
  const code='await (async()=>{const fs=await import("node:fs/promises"),crypto=await import("node:crypto");const file='+JSON.stringify(full)+';const real=(await fs.realpath(file)).replaceAll("\\\\","/").toLowerCase();if(!real.startsWith('+JSON.stringify(ROOT.toLowerCase()+'/')+'))throw Error("File escapes project");const h=await fs.open(file,"r"),b=Buffer.alloc('+item.bytes+');try{const r=await h.read(b,0,b.length,'+item.offset+');if(r.bytesRead!==b.length)throw Error("Short read");nodeRepl.write(JSON.stringify({base64:b.toString("base64"),sha:crypto.createHash("sha1").update("blob "+b.length+"\\0").update(b).digest("hex"),sha256:crypto.createHash("sha256").update(b).digest("hex")}));}finally{await h.close();}})();';
  const read=await tools.mcp__node_repl__js({code,title:'Transferring verified result bytes'});if(read.isError)throw Error('Artifact read failed');const payload=JSON.parse(read.content.find(x=>x.type==='text').text);
  if(payload.sha!==item.sha||payload.sha256!==item.sha256||payload.base64.length!==Math.ceil(item.bytes/3)*4)throw Error('Local file changed or transfer was truncated');
  const created=data(await tools.mcp__codex_apps__github_create_blob({repository_full_name:repo,encoding:'base64',content:payload.base64}));if(created.sha!==item.sha)throw Error('GitHub blob hash mismatch');existing.add(item.sha);uploaded++;bytes+=item.bytes;
  // Pace content writes below a request per second; standard API failures remain visible.
  await new Promise(resolve=>setTimeout(resolve,1200));
 }
 completed++;
 if(uploaded&&uploaded%10===0)await persist('GitBlobCache.json',{done:[...existing]});
 if(Date.now()-lastNotice>40000){notify('Prepared '+completed+'/'+plan.blobs.length+' Git objects; uploaded '+(bytes/2**20).toFixed(1)+' MiB. Branch remains unchanged.');lastNotice=Date.now();}
}
await persist('GitBlobCache.json',{done:[...existing]});notify('All result objects uploaded and hash-verified. Publishing the audited snapshot.');
const tree=data(await tools.mcp__codex_apps__github_create_tree({repository_full_name:repo,base_tree_sha:commit.tree.sha,tree_elements:plan.tree_elements}));
const made=data(await tools.mcp__codex_apps__github_create_commit({repository_full_name:repo,parent_sha:ref.object.sha,tree_sha:tree.sha,message:'Run source-aware learning and five-hour TinyStories training\n\nPublish controlled source-family results, fresh five-hour training, gated story-rehearsed QA, verified checkpoints, tokenizer and reproducible restoration tools. Binary files use hash-verified parts; logs and prior efficiency findings are archived. Existing local research history is retained in the history bundle.'}));
await tools.mcp__codex_apps__github_update_ref({repository_full_name:repo,branch_name:'main',sha:made.sha,expected_sha:ref.object.sha,force:false});
const after=data(await tools.mcp__codex_apps__github_fetch({url:api+'/git/ref/heads/main'}));if(after.object.sha!==made.sha)throw Error('Published branch head did not verify');
const published=data(await tools.mcp__codex_apps__github_fetch({url:api+'/git/trees/'+tree.sha+'?recursive=1'}));if(published.truncated)throw Error('Published tree verification truncated');const actual=new Map(published.tree.map(e=>[e.path,e.sha]));
for(const entry of plan.tree_elements){if(entry.sha===null){if(actual.has(entry.path))throw Error('Deletion did not verify');}else if(actual.get(entry.path)!==entry.sha)throw Error('Published file hash mismatch: '+entry.path);}
const receipt={status:'PASS',repository:repo,branch:'main',commit_sha:made.sha,tree_sha:tree.sha,parent_sha:ref.object.sha,url:'https://github.com/'+repo+'/commit/'+made.sha,verified_tree_entries:plan.tree_elements.length,binary_files:plan.binary_files,binary_bytes:plan.binary_bytes,transport:'Authenticated GitHub app; native Git credentials unavailable; byte-exact artifact parts with restoration manifest.'};
await persist('GitPublication_Receipt.json',receipt);text(receipt);
